/**
 * Incremental Gradient Descent Linear Model
 * EXACT port from IncGDLinearModel.java in the TAMER project
 */
export class LinearModel {
    /**
     * @param {number} numFeatures - Number of input features
     * @param {Object} options - Configuration options
     */
    constructor(numFeatures, options = {}) {
        this.numFeatures = numFeatures;

        // Learning parameters - defaults from Java Params.java line 55
        this.stepSize = options.stepSize ?? 0.05;
        this.regL2Wt = options.regL2Wt ?? 0.0;

        // Bias weight
        this.useBiasWt = options.useBiasWt ?? false;
        this.biasWt = options.initValue ?? 0.0;

        // Initialize weights (as in Java constructor)
        const initValue = options.initValue ?? 0.0;
        this.weights = new Array(numFeatures).fill(initValue);
        this.complSampleWts = new Array(numFeatures).fill(initValue);
        this.complSampleBiasWt = initValue;

        // Eligibility traces - EXACT from Java
        this.traces = new Array(numFeatures).fill(0);
        this.decayFactor = options.decayFactor ?? 0.0; // Default 0 disables traces
        this.discountFactor = options.discountFactor ?? 1.0;
        this.traceStyle = options.traceStyle ?? 'replacing';

        // Feature generator reference (set by agent)
        this.featGen = null;

        // Training statistics
        this.numUpdates = 0;
    }

    /**
     * Set the feature generator for predictForAction
     */
    setFeatGen(featGen) {
        this.featGen = featGen;
    }

    /**
     * Predict for a specific observation and action
     * Uses the feature generator to get features, then predicts
     */
    predictForAction(obs, action) {
        if (!this.featGen) {
            throw new Error('Feature generator not set. Call setFeatGen() first.');
        }
        const feats = this.featGen.getStateActionFeatures(obs, action);
        return this.predict(feats);
    }

    /**
     * Predict the output for a feature vector - EXACT port of predictLabel()
     * @param {number[]} sampleFeats - Feature vector
     * @returns {number} Predicted value
     */
    predict(sampleFeats) {
        let prediction = 0.0;
        for (let i = 0; i < this.weights.length; i++) {
            prediction += this.weights[i] * sampleFeats[i];
        }
        if (this.useBiasWt) {
            prediction += this.biasWt;
        }
        return prediction;
    }

    /**
     * Update eligibility traces - EXACT port of updateEligTraces()
     */
    _updateEligTraces(feats) {
        for (let i = 0; i < feats.length; i++) {
            // Decay previous traces
            this.traces[i] *= this.decayFactor * this.discountFactor;

            // Update traces with feats
            if (this.decayFactor * this.discountFactor === 0.0 || this.traceStyle === 'accumulating') {
                this.traces[i] += feats[i];
            } else if (this.traceStyle === 'replacing') {
                this.traces[i] = Math.max(feats[i], this.traces[i]);
            }
        }
    }

    /**
     * Gradient descent update - EXACT port of gradDescUpdate()
     */
    _gradDescUpdate(feats, label, sampleWeight, predictionAugmentation = 0) {
        const prediction = this.predict(feats) + predictionAugmentation;
        this._updateEligTraces(feats);
        const err = label - prediction;
        const wtForErr = this.stepSize * sampleWeight;

        for (let i = 0; i < this.weights.length; i++) {
            const wtedErr = wtForErr * (err - (this.regL2Wt * this.weights[i]));
            this.weights[i] += this.traces[i] * wtedErr;
        }

        if (this.useBiasWt) {
            this.biasWt += wtForErr * (err - (this.regL2Wt * this.complSampleBiasWt));
        }

        this.numUpdates++;
    }

    /**
     * Add a single instance - EXACT port of addInstance()
     * @param {number[]} feats - Feature vector
     * @param {number} label - Target value
     * @param {number} sampleWeight - Weight for this sample (default 1.0)
     */
    addInstance(feats, label, sampleWeight = 1.0) {
        // In Java: this.weights = this.complSampleWts
        // This copies reference, we'll copy values
        for (let i = 0; i < this.numFeatures; i++) {
            this.weights[i] = this.complSampleWts[i];
        }
        this.biasWt = this.complSampleBiasWt;

        this._gradDescUpdate(feats, label, sampleWeight, 0);

        // Update complSampleWts to match weights
        for (let i = 0; i < this.numFeatures; i++) {
            this.complSampleWts[i] = this.weights[i];
        }
        this.complSampleBiasWt = this.biasWt;
    }

    /**
     * Batch update with multiple samples
     * @param {Array<{features: number[], target: number, weight: number}>} samples
     */
    batchUpdate(samples) {
        for (const sample of samples) {
            this.addInstance(sample.features, sample.target, sample.weight ?? 1.0);
        }
    }

    /**
     * Add instances with replacement - EXACT port of addInstancesWReplacement()
     *
     * This is CRITICAL for credit assignment extrapolation to work correctly.
     *
     * Adds complete samples to a base model, copies that model to a temporary model
     * for use, and adds incomplete samples to the temporary model.
     *
     * - Complete samples (usedCredit > APPROX_ONE): permanently update complSampleWts
     * - Incomplete samples (usedCredit <= APPROX_ONE): temporarily update weights only
     *
     * @param {Array<{feats: number[], label: number, weight: number, usedCredit: number, unique: number}>} samples
     */
    addInstancesWReplacement(samples) {
        const APPROX_ONE = 0.99999;

        // Add all completed instances to base model
        for (const sample of samples) {
            if (sample.usedCredit > APPROX_ONE || sample.unique === -1) {
                // Complete sample: add to permanent base model
                this.addInstance(sample.feats, sample.label, sample.weight);
            }
        }

        // Copy base model to weights for predictions
        for (let i = 0; i < this.numFeatures; i++) {
            this.weights[i] = this.complSampleWts[i];
        }
        this.biasWt = this.complSampleBiasWt;

        // Add unfinished samples to copy (temporary, will be overwritten next step)
        for (const sample of samples) {
            if (sample.usedCredit <= APPROX_ONE && sample.weight !== 0 && sample.unique !== -1) {
                if (this.decayFactor !== 0.0) {
                    console.error('LinearModel does not support both eligibility traces and temporary samples.');
                }
                // This only updates this.weights, not complSampleWts
                this._gradDescUpdate(sample.feats, sample.label, sample.weight, 0);
            }
        }
    }

    /**
     * Clear traces
     */
    clearTraces() {
        this.traces.fill(0);
    }

    /**
     * Reset traces - alias for clearTraces (for SARSA compatibility)
     */
    resetTraces() {
        this.traces.fill(0);
    }

    /**
     * Build model after adding instances (no-op for incremental learning)
     * Provided for compatibility with ImitationAgent
     */
    buildModel() {
        // No-op for incremental gradient descent
    }

    /**
     * Set the step size (learning rate)
     */
    setStepSize(stepSize) {
        this.stepSize = stepSize;
    }

    /**
     * Set L2 regularization weight
     */
    setRegL2Wt(weight) {
        this.regL2Wt = weight;
    }

    /**
     * Get the current weights (for testing/debugging)
     */
    getWeights() {
        return [...this.weights];
    }

    /**
     * Set weights directly (for testing)
     */
    setWeights(weights) {
        this.weights = [...weights];
        this.complSampleWts = [...weights];
    }

    /**
     * Reset the model
     */
    reset() {
        this.weights.fill(0);
        this.complSampleWts.fill(0);
        this.traces.fill(0);
        this.biasWt = 0;
        this.complSampleBiasWt = 0;
        this.numUpdates = 0;
    }
}
