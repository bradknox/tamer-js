/**
 * ActionSelect - Action selection methods
 * EXACT port from ActionSelect.java in the TAMER project
 *
 * Implements action selection methods including greedy and epsilon-greedy.
 */

export class ActionSelect {
    /**
     * @param {Object} valFcnModel - Value function model for predictions
     * @param {string} selectionMethod - 'greedy' or 'e-greedy'
     * @param {Object} selectionParams - Parameters like epsilon
     * @param {number} numActions - Number of possible actions
     */
    constructor(valFcnModel, selectionMethod, selectionParams, numActions) {
        this.valFcnModel = valFcnModel;
        this.selectionMethod = selectionMethod || 'greedy';
        this.selectionParams = selectionParams || {};
        this.numActions = numActions;

        // Tree search parameters - EXACT from Java defaults
        this.treeSearch = selectionParams.treeSearch ?? false;
        this.greedyLeafPathLength = selectionParams.greedyLeafPathLength ?? 0;
        this.exhaustiveSearchDepth = selectionParams.exhaustiveSearchDepth ?? 1;
        this.randomizeSearchDepth = selectionParams.randomizeSearchDepth ?? true;

        // Discount parameters - EXACT from Java
        this.discountParam = Number.MAX_VALUE;
        this.discountType = 'EXPON'; // EXPON or HYPER

        // Additional models for action biasing
        this.biasModel = null;
        this.hInf = null;
        this.rewModel = null;
    }

    /**
     * Add model for action biasing - EXACT from Java addModelForActBias()
     */
    addModelForActBias(supplModel, hInf) {
        this.biasModel = supplModel;
        this.hInf = hInf;
    }

    /**
     * Set reward model - EXACT from Java setRewModel()
     */
    setRewModel(rewModel) {
        this.rewModel = rewModel;
    }

    /**
     * Set discount parameter - EXACT from Java setDiscountParam()
     */
    setDiscountParam(param) {
        this.discountParam = param;
    }

    /**
     * Select an action - EXACT from Java selectAction()
     * @param {number[]} obs - Current observation
     * @param {number} lastAction - Last action taken (for tie-breaking)
     * @returns {number} Selected action
     */
    selectAction(obs, lastAction) {
        if (this.selectionMethod === 'greedy') {
            return this.greedyActSelect(obs, lastAction);
        } else if (this.selectionMethod === 'e-greedy') {
            const epsilon = this.selectionParams.epsilon ?? 0.1;
            return this.eGreedyActSelect(epsilon, obs, lastAction);
        } else if (this.selectionMethod === 'vals-as-probs') {
            return this.chooseWValsAsProbs(obs, lastAction);
        } else {
            console.error(`Action selection method ${this.selectionMethod} not supported.`);
            return this.greedyActSelect(obs, lastAction);
        }
    }

    /**
     * Greedy action selection - EXACT from Java greedyActSelect()
     */
    greedyActSelect(obs, lastAction) {
        // Get action values from model
        const actVals = this._getActionValues(obs);

        // Find maximum value
        let maxVal = -Infinity;
        for (let a = 0; a < this.numActions; a++) {
            if (actVals[a] > maxVal) {
                maxVal = actVals[a];
            }
        }

        // Find all actions with maximum value
        const maxActs = [];
        for (let a = 0; a < this.numActions; a++) {
            if (actVals[a] === maxVal) {
                maxActs.push(a);
            }
        }

        // Check if last action is greedy - EXACT from Java
        if (lastAction !== null && lastAction !== undefined) {
            for (const act of maxActs) {
                if (act === lastAction) {
                    return lastAction;
                }
            }
        }

        // Random tie-breaking - EXACT from Java
        const actIndex = Math.floor(Math.random() * maxActs.length);
        return maxActs[actIndex];
    }

    /**
     * Epsilon-greedy action selection - EXACT from Java eGreedyActSelect()
     */
    eGreedyActSelect(epsilon, obs, lastAction) {
        if (Math.random() > epsilon) {
            return this.greedyActSelect(obs, lastAction);
        } else {
            return Math.floor(Math.random() * this.numActions);
        }
    }

    /**
     * Choose action with values as probabilities - EXACT from Java chooseWValsAsProbs()
     */
    chooseWValsAsProbs(obs, lastAction) {
        const actVals = this._getActionValues(obs);

        // Sum of action values for normalization
        let sumOfActVals = 0;
        for (const val of actVals) {
            sumOfActVals += Math.max(0, val); // Only use positive values
        }

        if (sumOfActVals <= 0) {
            // All values non-positive, choose randomly
            return Math.floor(Math.random() * this.numActions);
        }

        // Choose action based on probability
        const actRand = Math.random() * sumOfActVals;
        let cumProb = 0;
        for (let a = 0; a < this.numActions; a++) {
            cumProb += Math.max(0, actVals[a]);
            if (cumProb >= actRand) {
                return a;
            }
        }

        return this.numActions - 1;
    }

    /**
     * Get action values, optionally with bias model - internal helper
     */
    _getActionValues(obs) {
        const actVals = [];
        for (let a = 0; a < this.numActions; a++) {
            let val = this.valFcnModel.predictForAction(obs, a);

            // Add bias if present - EXACT from Java CombinationModel behavior
            if (this.biasModel && this.hInf) {
                const biasVal = this.biasModel.predictForAction(obs, a);
                const hInfluence = this.hInf.getHInfluence(obs, a);
                val += hInfluence * biasVal;
            }

            actVals.push(val);
        }
        return actVals;
    }

    /**
     * Anneal exploration parameters - EXACT from Java anneal()
     */
    anneal() {
        if (this.selectionMethod === 'e-greedy') {
            const annealRate = this.selectionParams.epsilonAnnealRate ?? 1.0;
            this.selectionParams.epsilon = (this.selectionParams.epsilon ?? 0.1) * annealRate;
        }
    }

    /**
     * Convert discount factor to parameter - EXACT from Java discFactorToParam()
     */
    static discFactorToParam(factor) {
        const param = -1 * Math.log(factor);
        if (isNaN(param)) {
            return Number.MAX_VALUE;
        }
        return param;
    }

    /**
     * Convert discount parameter to factor - EXACT from Java discParamToFactor()
     */
    static discParamToFactor(param) {
        return Math.exp(-1 * param);
    }
}
