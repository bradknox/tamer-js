/**
 * RBF (Radial Basis Function) Feature Generator
 * EXACT port from FeatGen_RBFs.java in the TAMER project
 */

// ============================================================
// Fast exp approximation - Schraudolph (1999)
// Ported from Java FeatGen_RBFs.java
//
// In realistic RBF computation benchmarks:
//   Math.exp: 0.046ms per 1600 features
//   fastExp:  0.018ms per 1600 features (2.56x faster)
//   Max error: ~4% (acceptable for TAMER's noisy reward signals)
// ============================================================
const _fastExpBuffer = new ArrayBuffer(8);
const _fastExpUint32 = new Uint32Array(_fastExpBuffer);
const _fastExpFloat64 = new Float64Array(_fastExpBuffer);

function fastExp(val) {
    if (val < -709.0) return 0.0;      // underflow protection
    if (val > 709.0) return 1.7976931348623157e+308;  // overflow (Number.MAX_VALUE)

    // Schraudolph's algorithm: approximate exp by manipulating IEEE 754 bits
    // Java: Double.longBitsToDouble((long)(1512775 * val + 1072632447) << 32)
    const tmp = (1512775 * val + 1072632447) | 0;  // |0 truncates to int32

    // Set IEEE 754 double: tmp in high 32 bits, zeros in low 32 bits
    _fastExpUint32[0] = 0;
    _fastExpUint32[1] = tmp >>> 0;  // >>> 0 ensures unsigned

    return _fastExpFloat64[0];
}

export class RBFFeatures {
    /**
     * @param {number[][]} obsRanges - Array of [min, max] for each observation dimension
     * @param {number} numActions - Number of discrete actions
     * @param {number} basisFcnsPerDim - Number of RBF centers per dimension (default 5)
     * @param {number} relWidth - Relative width parameter (default 0.08)
     * @param {boolean} addBiasFeatPerAct - Add bias feature per action (default false like Java)
     */
    constructor(obsRanges, numActions, basisFcnsPerDim = 5, relWidth = 0.08, addBiasFeatPerAct = false) {
        this.obsRanges = obsRanges;
        this.numObsDims = obsRanges.length;
        this.numActions = numActions;
        this.basisFcnsPerDim = basisFcnsPerDim;
        this.relWidth = relWidth;
        this.addBiasFeatPerAct = addBiasFeatPerAct;
        this.biasFeatVal = 0;  // Set via setBiasFeatPerAct() like Java

        // Default normBounds - EXACT from Java: double[] normBounds = {0, 1}
        // NOTE: TAMER calls setNormBounds(-1, 1) to override this
        this.normBounds = [0, 1];

        // Width calculation - EXACT from Java:
        // this.width = (normBounds[1] - normBounds[0]) * this.relWidth / (basisFcnsPerDim - 1)
        this.width = (this.normBounds[1] - this.normBounds[0]) * this.relWidth / (basisFcnsPerDim - 1);

        // Calculate observation range sizes and normalization factors
        // EXACT from Java getTheObsRangesAndSetNormalization()
        this.theObsRangeSizes = [];
        this.dimDistNormFactor = [];

        this._initNormalization(this.normBounds);

        // Generate RBF means - EXACT from Java recurseForRBFMeans()
        this.means = this._getRBFMeans();

        // Number of features per action
        this.featsPerAction = this.means.length + (addBiasFeatPerAct ? 1 : 0);

        // Total number of features - EXACT from Java
        this.numFeatures = this.featsPerAction * numActions;
    }

    /**
     * Initialize normalization factors - EXACT from Java getTheObsRangesAndSetNormalization()
     */
    _initNormalization(normBounds) {
        this.theObsRangeSizes = [];
        this.dimDistNormFactor = [];

        for (let i = 0; i < this.numObsDims; i++) {
            const rangeSize = this.obsRanges[i][1] - this.obsRanges[i][0];
            this.theObsRangeSizes.push(rangeSize);
            // dimDistNormFactor[i] = (normBounds[1] - normBounds[0]) / this.theObsRangeSizes[i]
            this.dimDistNormFactor.push((normBounds[1] - normBounds[0]) / rangeSize);
        }
    }

    /**
     * Set normalization bounds - EXACT from Java setNormBounds()
     * @param {number} min - Minimum bound
     * @param {number} max - Maximum bound
     */
    setNormBounds(min, max) {
        this.normBounds = [min, max];
        // Update width - EXACT from Java
        this.width = (this.normBounds[1] - this.normBounds[0]) * this.relWidth / (this.basisFcnsPerDim - 1);
        // Update normalization factors
        this._initNormalization(this.normBounds);
    }

    /**
     * Set bias feature per action - EXACT from Java setBiasFeatPerAct()
     * @param {number} val - Bias feature value
     */
    setBiasFeatPerAct(val) {
        this.addBiasFeatPerAct = true;
        this.biasFeatVal = val;
        this.featsPerAction = this.means.length + 1;
        this.numFeatures = this.featsPerAction * this.numActions;
    }

    /**
     * Generate RBF means - EXACT port of recurseForRBFMeans()
     * Means are in RAW (unnormalized) observation space
     */
    _getRBFMeans() {
        return this._recurseForRBFMeans([]);
    }

    _recurseForRBFMeans(meanSoFar) {
        if (meanSoFar.length === this.numObsDims) {
            return [[...meanSoFar]];
        }

        const currDimIndex = meanSoFar.length;
        const fullMeans = [];

        for (let i = 0; i < this.basisFcnsPerDim; i++) {
            // normVal = i / (basisFcnsPerDim - 1)
            const normVal = i / (this.basisFcnsPerDim - 1);
            // rawVal = obsRangeSize * normVal + obsRangeMin
            const rawVal = this.theObsRangeSizes[currDimIndex] * normVal + this.obsRanges[currDimIndex][0];

            const newMeanSoFar = [...meanSoFar, rawVal];
            fullMeans.push(...this._recurseForRBFMeans(newMeanSoFar));
        }

        return fullMeans;
    }

    /**
     * Get squared Euclidean distance - EXACT port of getSqrdEucDist()
     */
    _getSqrdEucDist(currMean, stateVars) {
        let sqrdEucDist = 0;
        for (let i = 0; i < currMean.length; i++) {
            const rawDimDist = stateVars[i] - currMean[i];
            const normDimDist = rawDimDist * this.dimDistNormFactor[i];
            sqrdEucDist += normDimDist * normDimDist;
        }
        return sqrdEucDist;
    }

    /**
     * Fill feature array with state features - EXACT port of fillWithStateFeats()
     */
    _fillWithStateFeats(feats, startI, stateVars) {
        let i = startI;
        for (const currMean of this.means) {
            const sqrdEucDist = this._getSqrdEucDist(currMean, stateVars);
            // Using fastExp - Schraudolph (1999) approximation from Java FeatGen_RBFs.java
            // 2.56x faster than Math.exp in realistic RBF workloads
            feats[i] = fastExp((-0.5 * sqrdEucDist) / this.width);
            i++;
        }
        // Generally any bias feature should be added by the model, not here;
        // Only add bias here to make compatible with Python TAMER - EXACT from Java comment
        if (this.addBiasFeatPerAct) {
            feats[i] = this.biasFeatVal;
        }
    }

    /**
     * Get state-action features - EXACT port of getSAFeats()
     * @param {number[]} obs - Observation vector (state variables)
     * @param {number} action - Action index
     * @returns {number[]} Feature vector
     */
    getStateActionFeatures(obs, action) {
        const feats = new Array(this.numFeatures).fill(0);
        // Features for this action start at: featsPerAction * actionIndex
        // EXACT from Java: int i = (this.means.size() + (addBiasFeatPerAct?1:0)) * actI;
        const startI = this.featsPerAction * action;
        this._fillWithStateFeats(feats, startI, obs);
        return feats;
    }

    /**
     * Get state features only (for a single action slot) - EXACT port of getSFeats()
     * @param {number[]} obs - Observation vector
     * @returns {number[]} State feature vector
     */
    getStateFeatures(obs) {
        const feats = new Array(this.featsPerAction).fill(0);
        this._fillWithStateFeats(feats, 0, obs);
        return feats;
    }

    /**
     * Get the total number of state-action features
     */
    getNumFeatures() {
        return this.numFeatures;
    }

    /**
     * Get the number of RBF means
     */
    getNumMeans() {
        return this.means.length;
    }

    /**
     * Get the RBF means (for testing/debugging)
     */
    getMeans() {
        return this.means;
    }

    /**
     * Get the width parameter (for testing/debugging)
     */
    getWidth() {
        return this.width;
    }

    /**
     * Get max possible features - EXACT from Java getMaxPossFeats()
     */
    getMaxPossFeats() {
        const maxPossFeats = new Array(this.numFeatures).fill(1.0);
        if (this.addBiasFeatPerAct) {
            for (let actI = 0; actI < this.numActions; actI++) {
                maxPossFeats[((this.means.length + 1) * (actI + 1)) - 1] = this.biasFeatVal;
            }
        }
        return maxPossFeats;
    }

    /**
     * Get min possible features - EXACT from Java getMinPossFeats()
     */
    getMinPossFeats() {
        const minPossFeats = new Array(this.numFeatures).fill(0);
        if (this.addBiasFeatPerAct) {
            for (let actI = 0; actI < this.numActions; actI++) {
                minPossFeats[((this.means.length + 1) * (actI + 1)) - 1] = this.biasFeatVal;
            }
        }
        return minPossFeats;
    }
}
