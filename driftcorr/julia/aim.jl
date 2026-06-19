#!/usr/bin/env julia
# =============================================================================
# Faithful Julia port of Adaptive Intersection Maximization (AIM) drift correction.
#
# Translated line-by-line from the authors' MATLAB reference implementation:
#   AIM/AIM.m and AIM/IntersectionMax.m
#   Hongqiang Ma, University of Pittsburgh (2023) / UIUC (2024)
#   https://github.com/YangLiuLab/AIM
#
# Citation: Hongqiang Ma, Maomao Chen, Phuong Nguyen, Yang Liu,
#   "Toward drift-free high-throughput nanoscopy through adaptive intersection
#   maximization", Sci. Adv. 10, eadm7765 (2024). DOI:10.1126/sciadv.adm7765
#
# This is an independent translation for use in SMLM LabFlow. The upstream repo
# ships no explicit license; this port is provided for research use with the
# citation above. The numerics mirror the MATLAB: 1D-index grid at pitch
# IntersectD, 7x7 coincidence ROI, sub-pixel peak from the phase of the first
# Fourier harmonics (fft2(ROIcc') coefficients (1,2)/(2,1)), sequential
# pre-shift tracking with a fixed reference, two rounds, and a cubic-spline
# (not-a-knot) interpolation of per-segment drift to per-frame.
#
# Stdlib only (DelimitedFiles, LinearAlgebra, Printf) -- no package install.
# =============================================================================

using DelimitedFiles
using LinearAlgebra
using Printf

const ROIR = 3
const ROI_SIZE = 2 * ROIR + 1

mround(x) = round(x, RoundNearestTiesAway)              # MATLAB round() semantics
mround_int(x) = round(Int, x, RoundNearestTiesAway)


# --- not-a-knot cubic spline (matches MATLAB interp1(...,'spline')) ----------
function spline_natknot(x::Vector{Float64}, y::Vector{Float64}, xq::Vector{Float64})
    n = length(x)
    h = diff(x)
    A = zeros(n, n)
    b = zeros(n)
    # interior moment equations
    for i in 2:n-1
        A[i, i-1] = h[i-1]
        A[i, i]   = 2 * (h[i-1] + h[i])
        A[i, i+1] = h[i]
        b[i] = 6 * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1])
    end
    # not-a-knot end conditions (third derivative continuous at x[2], x[n-1])
    A[1, 1] = h[2];            A[1, 2] = -(h[1] + h[2]);     A[1, 3] = h[1]
    A[n, n-2] = h[n-1];        A[n, n-1] = -(h[n-2] + h[n-1]); A[n, n] = h[n-2]
    M = A \ b
    # evaluate
    out = similar(xq)
    @inbounds for k in eachindex(xq)
        q = xq[k]
        i = searchsortedlast(x, q)
        i = clamp(i, 1, n - 1)
        hi = h[i]
        a1 = x[i+1] - q
        a2 = q - x[i]
        out[k] = M[i] * a1^3 / (6hi) + M[i+1] * a2^3 / (6hi) +
                 (y[i] / hi - M[i] * hi / 6) * a1 +
                 (y[i+1] / hi - M[i+1] * hi / 6) * a2
    end
    return out
end


# --- IntersectionMax (faithful translation of IntersectionMax.m) -------------
# Returns per-frame drift (length frameMax) in coordinate units.
function intersection_max(XList, YList, refXList, refYList, fID,
                          trackNUM, trackInterval, stride, IntersectD, frameMax)
    encode(X, Y) = mround_int.(Y ./ IntersectD) .* stride .+ mround_int.(X ./ IntersectD)
    pList = encode(XList, YList)
    refList = encode(refXList, refYList)

    # reference sparse histogram: position -> count
    refcount = Dict{Int,Float64}()
    for p in refList
        refcount[p] = get(refcount, p, 0.0) + 1.0
    end

    # per-segment sparse histograms (one pass over all localizations)
    seg_hist = [Dict{Int,Float64}() for _ in 1:trackNUM]
    @inbounds for k in eachindex(fID)
        s = cld(fID[k], trackInterval)          # segment index = ceil(f/interval)
        if 1 <= s <= trackNUM
            d = seg_hist[s]
            p = pList[k]
            d[p] = get(d, p, 0.0) + 1.0
        end
    end

    W = exp(-2pi * im / ROI_SIZE)
    driftX = zeros(trackNUM)
    driftY = zeros(trackNUM)
    refx = 0.0
    refy = 0.0

    for s in 2:trackNUM
        seg = seg_hist[s]
        sft = mround_int(refy) * stride + mround_int(refx)

        # 7x7 intersection (coincidence) map; ROIcc[c+ROIR+1, r+ROIR+1]
        ROIcc = zeros(Float64, ROI_SIZE, ROI_SIZE)
        for (p, v) in seg
            pp = p + sft
            for r in -ROIR:ROIR
                base = pp + r * stride
                for c in -ROIR:ROIR
                    rc = get(refcount, base + c, 0.0)
                    if rc != 0.0
                        ROIcc[c+ROIR+1, r+ROIR+1] += v * rc
                    end
                end
            end
        end

        # sub-pixel peak from the phase of fft2(ROIcc') coefficients (1,2),(2,1)
        F12 = 0.0 + 0.0im
        F21 = 0.0 + 0.0im
        for j in 1:ROI_SIZE
            rs = 0.0
            for i in 1:ROI_SIZE
                rs += ROIcc[j, i]
            end
            F12 += rs * W^(j - 1)
        end
        for i in 1:ROI_SIZE
            cs = 0.0
            for j in 1:ROI_SIZE
                cs += ROIcc[j, i]
            end
            F21 += cs * W^(i - 1)
        end
        angX = angle(F12); angX -= 2pi * (angX > 0)
        PX = (abs(angX) / (2pi / ROI_SIZE) + 1) - (ROI_SIZE + 1) / 2
        angY = angle(F21); angY -= 2pi * (angY > 0)
        PY = (abs(angY) / (2pi / ROI_SIZE) + 1) - (ROI_SIZE + 1) / 2

        refx = mround(refx) + PX
        refy = mround(refy) + PY
        driftX[s] = -refx
        driftY[s] = -refy
    end

    # spline interpolation with linear end-padding, scaled to coordinate units
    dXpad = vcat(2 * driftX[1] - driftX[2], driftX, 2 * driftX[end] - driftX[end-1]) .* IntersectD
    dYpad = vcat(2 * driftY[1] - driftY[2], driftY, 2 * driftY[end] - driftY[end-1]) .* IntersectD
    xs = collect(-0.5:1.0:(trackNUM + 0.5)) .* trackInterval
    qf = collect(1.0:frameMax)
    dX = spline_natknot(xs, dXpad, qf)
    dY = spline_natknot(xs, dYpad, qf)
    return dX, dY
end


# --- AIM (faithful translation of AIM.m, 2D path) ----------------------------
function aim(F::Vector{Int}, X::Vector{Float64}, Y::Vector{Float64},
             trackInterval::Int, IntersectD::Float64)
    F = F .- minimum(F) .+ 1
    frameNUM = maximum(F)
    trackNUM = fld(frameNUM, trackInterval)
    frameNUM = trackNUM * trackInterval
    Fc = map(f -> f > frameNUM ? frameNUM : f, F)
    stride = mround_int(maximum(X) / IntersectD) + 50      # grid row stride (cells)

    refmask = Fc .<= trackInterval
    # round 1: reference = first segment
    dX1, dY1 = intersection_max(X, Y, X[refmask], Y[refmask], Fc,
                                trackNUM, trackInterval, stride, IntersectD, frameNUM)
    Xpdc = X .- dX1[Fc]
    Ypdc = Y .- dY1[Fc]
    # round 2: reference = entire round-1-corrected dataset
    dX2, dY2 = intersection_max(Xpdc, Ypdc, Xpdc, Ypdc, Fc,
                                trackNUM, trackInterval, stride, IntersectD, frameNUM)

    driftx = dX1 .+ dX2
    drifty = dY1 .+ dY2
    driftx .-= driftx[1]
    drifty .-= drifty[1]
    return driftx, drifty, frameNUM
end


# --- CLI ---------------------------------------------------------------------
function parse_args(args)
    d = Dict{String,String}()
    i = 1
    while i <= length(args)
        a = args[i]
        if startswith(a, "--")
            d[a[3:end]] = args[i+1]
            i += 2
        else
            i += 1
        end
    end
    return d
end

function main()
    opt = parse_args(ARGS)
    inpath = opt["in"]
    outpath = opt["out"]
    trackInterval = parse(Int, get(opt, "track-interval", "500"))
    IntersectD = parse(Float64, get(opt, "intersect", "20.0"))

    data, _ = readdlm(inpath, ',', header = true)
    F = Int.(round.(Float64.(data[:, 1])))
    X = Float64.(data[:, 2])
    Y = Float64.(data[:, 3])

    driftx, drifty, frameNUM = aim(F, X, Y, trackInterval, IntersectD)

    # map per-frame drift back to original (unsorted) unique frames
    fmin = minimum(F)
    uniq = sort(unique(F))
    open(outpath, "w") do io
        println(io, "frame,dx,dy")
        for f in uniq
            idx = clamp(f - fmin + 1, 1, frameNUM)
            @printf(io, "%d,%.6f,%.6f\n", f, driftx[idx], drifty[idx])
        end
    end
end

main()
