/**
 * HInfluence - Human reward model influence for TAMER+RL
 * EXACT port from HInfluence.java in the TAMER project
 *
 * Calculates a parameter (beta) that determines the relative influence
 * of the model of human reward for TAMER+RL.
 */
import { CreditAssign } from './CreditAssign.js';

export class HInfluence {
    /**
     * @param {string} influenceMethod - 'annealedParam' or 'eligTrace'
     * @param {number} combParam - Combination parameter (beta)
     * @param {Object} options - Additional options
     */
    constructor(influenceMethod, combParam, options = {}) {
        // EXACT from Java
        this.INFLUENCE_METHOD = influenceMethod || 'annealedParam';
        this.COMB_PARAM = combParam || 1.0;
        this.STEP_DECAY_FACTOR = options.stepDecayFactor ?? 1.0;
        this.EP_DECAY_FACTOR = options.epDecayFactor ?? 1.0;
        this.TRACE_STYLE = options.traceStyle || 'accumulating';
        this.ACCUM_FACTOR = options.accumFactor ?? 0.0;

        this.featGen = options.featGen || null;
        this.stateOnly = options.stateOnly ?? false;

        // Initialize traces - EXACT from Java
        if (this.INFLUENCE_METHOD === 'annealedParam') {
            this.traces = new Float64Array(1);
            this.setTracesToMax();
        } else if (this.INFLUENCE_METHOD === 'eligTrace' && this.featGen) {
            // Create credit assignment for eligibility trace method
            this.credA = new CreditAssign(options.creditAssignParams || {});

            // Initialize traces based on feature size
            const numFeats = this.stateOnly ?
                this.featGen.getNumStateFeatures() :
                this.featGen.getNumFeatures();

            this.traces = new Float64Array(numFeats);
            this.lastStepTraces = new Float64Array(numFeats);

            // Feature bounds for normalization
            this.minFeats = new Float64Array(numFeats).fill(0);
            this.maxFeats = new Float64Array(numFeats).fill(1);
        } else {
            this.traces = new Float64Array(1);
            this.setTracesToMax();
        }

        this.lastStepTraces = new Float64Array(this.traces.length);
    }

    /**
     * Set traces to maximum value - EXACT from Java setTracesToMax()
     */
    setTracesToMax() {
        for (let i = 0; i < this.traces.length; i++) {
            this.traces[i] = 1.0;
        }
    }

    /**
     * Set accumulation factor
     */
    setAccumFactor(val) {
        this.ACCUM_FACTOR = val;
    }

    /**
     * Set trace style
     */
    setTraceStyle(traceStyle) {
        this.TRACE_STYLE = traceStyle;
    }

    /**
     * Set step decay factor - EXACT from Java setStepDecayFactor()
     */
    setStepDecayFactor(stepDecayFactor) {
        this.STEP_DECAY_FACTOR = stepDecayFactor;
    }

    /**
     * Set episode decay factor - EXACT from Java setEpDecayFactor()
     */
    setEpDecayFactor(epDecayFactor) {
        this.EP_DECAY_FACTOR = epDecayFactor;
    }

    /**
     * Linear feature normalization - EXACT from Java linearNormFeats()
     */
    _linearNormFeats(feats) {
        const normFeats = new Float64Array(feats.length);
        for (let i = 0; i < feats.length; i++) {
            const range = this.maxFeats[i] - this.minFeats[i];
            if (range === 0) {
                normFeats[i] = 0.0;
            } else {
                normFeats[i] = (feats[i] - this.minFeats[i]) / range;
            }
        }
        return normFeats;
    }

    /**
     * Episode end update - EXACT from Java episodeEndUpdate()
     */
    episodeEndUpdate() {
        this.lastStepTraces = new Float64Array(this.traces);
        if (this.EP_DECAY_FACTOR !== 1.0) {
            this._epDecayEligTraces();
        }
    }

    /**
     * Get H influence for observation and action - EXACT from Java getHInfluence()
     */
    getHInfluence(obs, action, lastStepInf = false) {
        const tracesForInf = lastStepInf ? this.lastStepTraces : this.traces;

        if (this.INFLUENCE_METHOD === 'annealedParam') {
            return tracesForInf[0] * this.COMB_PARAM;
        }

        if (this.INFLUENCE_METHOD === 'eligTrace' && this.featGen) {
            const feats = this.stateOnly ?
                this.featGen.getStateFeatures(obs) :
                this.featGen.getStateActionFeatures(obs, action);
            return this._getHInfluenceFromFeats(feats, lastStepInf);
        }

        return tracesForInf[0] * this.COMB_PARAM;
    }

    /**
     * Get H influence from features - EXACT from Java getHInfluence(double[] feats)
     */
    _getHInfluenceFromFeats(feats, lastStepInf = false) {
        const tracesForInf = lastStepInf ? this.lastStepTraces : this.traces;

        if (this.INFLUENCE_METHOD === 'annealedParam') {
            return tracesForInf[0] * this.COMB_PARAM;
        }

        if (this.INFLUENCE_METHOD === 'eligTrace') {
            let eligAndFeatDotPr = 0;
            let normFeatsL1Norm = 0;
            const normFeats = this._linearNormFeats(feats);

            for (let i = 0; i < normFeats.length; i++) {
                eligAndFeatDotPr += normFeats[i] * Math.min(1.0, tracesForInf[i]);
                normFeatsL1Norm += normFeats[i];
            }

            if (normFeatsL1Norm === 0) return 0;

            const hInflWt = this.COMB_PARAM * (eligAndFeatDotPr / normFeatsL1Norm);
            return hInflWt;
        }

        return this.COMB_PARAM;
    }

    /**
     * Step update - EXACT from Java stepUpdate()
     */
    stepUpdate(inTrainSess, stepStartTime) {
        this.lastStepTraces = new Float64Array(this.traces);

        // Decay
        if (this.STEP_DECAY_FACTOR !== 1.0) {
            this._stepDecayEligTraces();
        }

        if (this.INFLUENCE_METHOD === 'eligTrace' && this.credA) {
            // Process samples from credit assignment
            const samples = this.credA.processSamplesAndRemoveFinished(stepStartTime, inTrainSess);

            if (inTrainSess) {
                for (const sample of samples) {
                    this._growEligTraces(sample.feats, sample.creditUsedLastStep || 1.0);
                }
            }
        }
    }

    /**
     * Record time step start - EXACT from Java recordTimeStepStart()
     */
    recordTimeStepStart(obs, action, startTime) {
        if (this.INFLUENCE_METHOD === 'eligTrace' && this.credA && this.featGen) {
            const feats = this.stateOnly ?
                this.featGen.getStateFeatures(obs) :
                this.featGen.getStateActionFeatures(obs, action);
            const normFeats = this._linearNormFeats(feats);
            this.credA.recordTimeStepStart(normFeats, startTime);
        }
    }

    /**
     * Record time step end - EXACT from Java recordTimeStepEnd()
     */
    recordTimeStepEnd(endTime) {
        if (this.INFLUENCE_METHOD === 'eligTrace' && this.credA) {
            this.credA.recordTimeStepEnd(endTime);
        }
    }

    /**
     * Step decay eligibility traces - EXACT from Java stepDecayEligTraces()
     */
    _stepDecayEligTraces() {
        for (let i = 0; i < this.traces.length; i++) {
            this.traces[i] *= this.STEP_DECAY_FACTOR;
        }
    }

    /**
     * Episode decay eligibility traces - EXACT from Java epDecayEligTraces()
     */
    _epDecayEligTraces() {
        for (let i = 0; i < this.traces.length; i++) {
            this.traces[i] *= this.EP_DECAY_FACTOR;
        }
    }

    /**
     * Grow eligibility traces - EXACT from Java growEligTraces()
     */
    _growEligTraces(normFeats, weight) {
        for (let i = 0; i < normFeats.length; i++) {
            if (this.TRACE_STYLE === 'replacing') {
                this.traces[i] = Math.max(normFeats[i], this.traces[i]);
            } else if (this.TRACE_STYLE === 'accumulating') {
                this.traces[i] = Math.min(
                    (weight * normFeats[i] * this.ACCUM_FACTOR) + this.traces[i],
                    1.0
                );
            }
        }
    }
}
