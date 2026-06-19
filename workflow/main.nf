#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

/*
 * SMLM LabFlow - registry-driven Nextflow pipeline (mirror of workflow/Snakefile).
 *
 *   calibrate -> train -> infer -> drift -> [track | cluster | spatial_stats |
 *                                            counting | phenotype | analyze] -> report
 *
 * Every stage shells `python -m labflow run <stage> --method <name>`, dispatched
 * through the SAME registry (config/methods.yaml) as the CLI and the Snakemake
 * pipeline -- so adding a method stays a single YAML entry. Stages operate in place
 * on params.run_dir (like the Snakefile), producing the identical output tree, so
 * generate_run_report.py and everything downstream are unchanged.
 *
 * This file is a faithful port of the verified Snakefile; validate an actual run on
 * the target machine (needs Java 11+ and the `nextflow` binary).
 */

import groovy.json.JsonOutput

// "--param k=v" for the chosen method of a stage, from params.<stage>_params (mirrors
// the Snakefile's param_args). JSON-encodes values so labflow's --param parser matches.
def paramArgs = { String stage, String method ->
    def all  = params.containsKey("${stage}_params") ? params."${stage}_params" : [:]
    def over = (all instanceof Map && all[method] instanceof Map) ? all[method] : [:]
    over.collect { k, v -> "--param ${k}=${JsonOutput.toJson(v)}" }.join(' ')
}

def pixelArg = ("${params.pixel_size_nm}".trim() && "${params.pixel_size_nm}" != 'None') ?
               "--pixel-size ${params.pixel_size_nm}" : ''

// ---- localization lifecycle (optional calibrate/train, then infer) -----------

process CALIBRATE {
    input:  val gate
    output: val 'calibrated'
    script:
    """
    cd ${params.repo}
    python run_pipeline.py calibrate -i "${params.calibrate_input}" -p "${params.profile}" \
        -o "${params.run_dir}/calibrate" -b "${params.backend}" --overwrite
    """
}

process TRAIN {
    input:  val gate
    output: val 'trained'
    script:
    """
    cd ${params.repo}
    python run_pipeline.py train -i "${params.train_input}" -p "${params.profile}" \
        -o "${params.run_dir}/train" -b "${params.backend}" --overwrite
    """
}

// Localization: the existing pipeline. A checkpoint in Snakemake -- here the batch
// set is read from the manifest it writes (staged out so the channel can fan out).
process INFER {
    input:  val gate
    output: path 'batch_manifest.csv'
    script:
    def res = "${params.run_dir}/infer/results"
    """
    cd ${params.repo}
    python run_pipeline.py infer -i "${params.input_dir}" -p "${params.profile}" \
        -o "${params.run_dir}/infer" -b "${params.backend}" --overwrite ${params.extra_infer_args}
    cp "${res}/batch_manifest.csv" batch_manifest.csv
    """
}

// ---- per-batch analysis stages (each = one `labflow run`) ---------------------

process DRIFT {
    tag "$batch"
    input:
        val batch
        val extra
    output:
        val batch
    script:
    def b = "${params.run_dir}/infer/results/batches/${batch}"
    """
    cd ${params.repo}
    python -m labflow run drift --method ${params.drift_method} \
        -i "${b}/canonical_localizations.csv" \
        -o "${b}/drift/drift_corrected_localizations.csv" ${extra}
    """
}

process TRACK {
    tag "$batch"
    input:
        val batch
        val extra
    output:
        val batch
    script:
    def b = "${params.run_dir}/infer/results/batches/${batch}"
    """
    cd ${params.repo}
    python -m labflow run track --method ${params.track_method} \
        -i "${b}/drift/drift_corrected_localizations.csv" -o "${b}/track/tracks.csv" ${extra}
    """
}

process CLUSTER {
    tag "$batch"
    input:
        val batch
        val extra
    output:
        val batch
    script:
    def b = "${params.run_dir}/infer/results/batches/${batch}"
    """
    cd ${params.repo}
    python -m labflow run cluster --method ${params.cluster_method} \
        -i "${b}/drift/drift_corrected_localizations.csv" -o "${b}/cluster/clusters.csv" ${extra}
    """
}

process SPATIAL_STATS {
    tag "$batch"
    input:
        val batch
        val extra
    output:
        val batch
    script:
    def b = "${params.run_dir}/infer/results/batches/${batch}"
    """
    cd ${params.repo}
    python -m labflow run spatial_stats --method ${params.spatial_stats_method} \
        -i "${b}/drift/drift_corrected_localizations.csv" -o "${b}/spatial_stats/spatial_stats.csv" ${extra}
    """
}

process COUNTING {
    tag "$batch"
    input:
        val batch
        val extra
    output:
        val batch
    script:
    def b = "${params.run_dir}/infer/results/batches/${batch}"
    """
    cd ${params.repo}
    python -m labflow run counting --method ${params.counting_method} \
        -i "${b}/cluster/clusters.csv" -o "${b}/counting/counts.csv" ${extra}
    """
}

process PHENOTYPE {
    tag "$batch"
    input:
        val batch
        val extra
    output:
        val batch
    script:
    def b = "${params.run_dir}/infer/results/batches/${batch}"
    """
    cd ${params.repo}
    python -m labflow run phenotype --method ${params.phenotype_method} \
        -i "${b}/cluster/clusters.csv" -o "${b}/phenotype/phenotypes.csv" ${extra}
    """
}

process ANALYZE {
    tag "$batch"
    input:
        val batch
        val extra
    output:
        val batch
    script:
    def b = "${params.run_dir}/infer/results/batches/${batch}"
    """
    cd ${params.repo}
    python -m labflow run analyze --method ${params.analyze_method} \
        -i "${b}/track/tracks.csv" -o "${b}/analyze/track_analysis.csv" ${extra}
    """
}

process REPORT {
    input: val done
    script:
    def res = "${params.run_dir}/infer/results"
    """
    cd ${params.repo}
    python generate_run_report.py --run "${res}"
    """
}

// ---- DAG --------------------------------------------------------------------

workflow {
    // optional lifecycle: chain gates so infer waits only for what is enabled
    gate = Channel.value('start')
    if (params.calibrate) { gate = CALIBRATE(gate) }
    if (params.train)     { gate = TRAIN(gate) }

    manifest = INFER(gate)
    batches  = manifest.splitCsv(header: true)
                       .map { row -> row.run_id ?: row.batch_id }
                       .filter { it }

    // constant per-stage arg strings (computed once; reused for every batch)
    drifted = DRIFT(batches, Channel.value("${pixelArg} --units ${params.units} " +
                                           paramArgs('drift', "${params.drift_method}")))
    done = drifted

    tracked   = params.track   ? TRACK(drifted,   Channel.value(paramArgs('track', "${params.track_method}")))     : Channel.empty()
    clustered = params.cluster ? CLUSTER(drifted, Channel.value(paramArgs('cluster', "${params.cluster_method}"))) : Channel.empty()
    if (params.track)   { done = done.mix(tracked) }
    if (params.cluster) { done = done.mix(clustered) }
    if (params.spatial_stats) { done = done.mix(SPATIAL_STATS(drifted,   Channel.value(paramArgs('spatial_stats', "${params.spatial_stats_method}")))) }
    if (params.counting)      { done = done.mix(COUNTING(clustered,      Channel.value(paramArgs('counting', "${params.counting_method}")))) }    // requires cluster
    if (params.phenotype)     { done = done.mix(PHENOTYPE(clustered,     Channel.value(paramArgs('phenotype', "${params.phenotype_method}")))) }  // requires cluster
    if (params.analyze)       { done = done.mix(ANALYZE(tracked,         Channel.value(paramArgs('analyze', "${params.analyze_method}")))) }      // requires track

    REPORT(done.collect())
}
