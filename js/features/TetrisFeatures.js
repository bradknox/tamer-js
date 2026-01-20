/**
 * TetrisFeatures - Specialized feature generator for Tetris
 * EXACT port from FeatGen_Tetris.java in the TAMER project
 *
 * Produces 46 hand-crafted features that capture meaningful Tetris state:
 * - Column heights (10)
 * - Max column height (1)
 * - Column height differences (9)
 * - Number of holes (1)
 * - Max well depth (1)
 * - Sum of well depths (1)
 * - Squared versions of above (23)
 */

import { TetrisState } from '../environments/tetris/TetrisState.js?v=9';

/**
 * ExtendedTetrisAction - Represents a piece placement (sequence of atomic actions)
 * EXACT from Java FeatGen_Tetris.ExtendedTetrisAction
 */
export class ExtendedTetrisAction {
    constructor(currentRotation, currentX, currentY, blockMobile, actList) {
        this.currentRotation = currentRotation;
        this.currentX = currentX;
        this.currentY = currentY;
        this.blockMobile = blockMobile;
        this.actList = actList;
    }

    setState(state) {
        state.currentRotation = this.currentRotation;
        state.currentX = this.currentX;
        state.currentY = this.currentY;
        state.blockMobile = this.blockMobile;
    }

    addNoneAction() {
        const newActList = new Array(this.actList.length + 1);
        for (let i = 0; i < this.actList.length; i++) {
            newActList[i] = this.actList[i];
        }
        newActList[newActList.length - 1] = 4; // NONE action
        this.actList = newActList;
    }
}

export class TetrisFeatures {
    // Feature indices - EXACT from Java
    static NUM_FEATS = 46;
    static COL_HT_START_I = 0;
    static MAX_COL_HT_I = 10;
    static COL_DIFF_START_I = 11;
    static NUM_HOLES_I = 20;
    static MAX_WELL_I = 21;
    static SUM_WELL_I = 22;
    static SQUARED_FEATS_START_I = 23;

    // Scaling factor for squared height features - EXACT from Java
    static HT_SQ_SCALE = 100.0;

    // Configuration flags - EXACT from Java
    static REMOVE_DECIMAL = false;
    static COUNT_LAST_COL_FOR_WELL_SUM = true;
    static SCALE_ALL_SQUARED_FEATS = false;

    /**
     * @param {Array} obsRanges - Observation ranges (unused, for interface compatibility)
     * @param {number} numActions - Number of actions (for atomic actions - not used for extended)
     */
    constructor(obsRanges, numActions) {
        this.numActions = numActions;
        // For getStateActionFeatures (non-extended): 46 * numActions
        // For getSAFeats (extended actions): just 46
        this.numFeatures = TetrisFeatures.NUM_FEATS * numActions;

        // Board dimensions from Tetris (10x20)
        this.worldWidth = 10;
        this.worldHeight = 20;

        // Internal TetrisState for simulating moves - EXACT from Java
        this.gameState = new TetrisState();

        // Temporary feature arrays - EXACT from Java
        this.currSFeats = new Float64Array(TetrisFeatures.NUM_FEATS);
        this.nextSFeats = new Float64Array(TetrisFeatures.NUM_FEATS);
    }

    /**
     * Get number of features (for getStateActionFeatures interface)
     */
    getNumFeatures() {
        return this.numFeatures;
    }

    /**
     * Get number of state features (without action)
     */
    getNumStateFeatures() {
        return TetrisFeatures.NUM_FEATS;
    }

    /**
     * Get number of features for extended actions (getSAFeats)
     * Used by TamerAgent when usesExtendedActions() is true
     */
    getNumExtendedFeatures() {
        return TetrisFeatures.NUM_FEATS;
    }

    /**
     * Set internal state from observation - EXACT from Java setStateFromObs()
     */
    setStateFromObs(obs) {
        for (let i = 0; i < this.gameState.worldState.length; i++) {
            this.gameState.worldState[i] = obs[i];
        }
        this.gameState.blockMobile = obs[this.gameState.worldState.length] === 1;
        this.gameState.currentBlockId = obs[this.gameState.worldState.length + 1];
        this.gameState.currentRotation = obs[this.gameState.worldState.length + 2];
        this.gameState.currentX = obs[this.gameState.worldState.length + 3];
        this.gameState.currentY = obs[this.gameState.worldState.length + 4];
        this.gameState.worldWidth = obs[this.gameState.worldState.length + 5];
        this.gameState.worldHeight = obs[this.gameState.worldState.length + 6];
    }

    /**
     * Get possible extended actions (piece placements) - EXACT from Java getPossActions()
     * @param {Array} obs - Current observation
     * @returns {Array<ExtendedTetrisAction>} List of possible piece placements
     */
    getPossActions(obs) {
        this.setStateFromObs(obs);
        return this.getExtendedActList(obs);
    }

    /**
     * Tree search to find all possible piece placements - EXACT from Java getExtendedActList()
     * @param {Array} intObsVals - Observation values
     * @returns {Array<ExtendedTetrisAction>} All possible placements
     */
    getExtendedActList(intObsVals) {
        const actOrder = [4, 0, 1, 2, 3];  // NONE, LEFT, RIGHT, CW, CCW - EXACT from Java
        const extendedActList = [];
        const tempState = new TetrisState(this.gameState);  // Copy world state

        const startMove = new ExtendedTetrisAction(
            tempState.currentRotation,
            tempState.currentX,
            tempState.currentY,
            tempState.blockMobile,
            []
        );

        let liveMoves = [startMove];

        // Search tree of possible moves, pruning duplicates
        while (liveMoves.length > 0) {
            const nextLevelLiveMoves = [];

            for (const actNum of actOrder) {
                for (const liveMove of liveMoves) {
                    liveMove.setState(tempState);

                    // Take action and skip if action was illegal
                    if (!tempState.takeAction(actNum)) {
                        continue;
                    }

                    tempState.update();

                    // If move is duplicate of finished extended action, continue
                    if (this._isInExtActList(tempState, extendedActList)) {
                        continue;
                    }

                    // If move is duplicate of live extended action, continue
                    if ((this._isInExtActList(tempState, nextLevelLiveMoves) ||
                         this._isInExtActList(tempState, liveMoves)) && tempState.blockMobile) {
                        continue;
                    }

                    // Make new extended action
                    const thisActList = new Array(liveMove.actList.length + 1);
                    for (let i = 0; i < liveMove.actList.length; i++) {
                        thisActList[i] = liveMove.actList[i];
                    }
                    thisActList[thisActList.length - 1] = actNum;

                    const thisMove = new ExtendedTetrisAction(
                        tempState.currentRotation,
                        tempState.currentX,
                        tempState.currentY,
                        tempState.blockMobile,
                        thisActList
                    );

                    if (tempState.blockMobile) {
                        // Block is active, add to live moves
                        nextLevelLiveMoves.push(thisMove);
                    } else {
                        // Move is inactive (piece placed), add to final list
                        extendedActList.push(thisMove);
                    }
                }
            }

            liveMoves = nextLevelLiveMoves;
        }

        // Add a no-action to each placement - EXACT from Java
        for (const extAct of extendedActList) {
            extAct.addNoneAction();
        }

        return extendedActList;
    }

    /**
     * Check if state matches any extended action in list - EXACT from Java isInExtActList()
     */
    _isInExtActList(state, moves) {
        for (const move of moves) {
            if (state.equivRotation(move.currentRotation) &&
                state.currentX === move.currentX &&
                state.currentY === move.currentY) {
                return true;
            }
        }
        return false;
    }

    /**
     * Get state-action features (state change features) - EXACT from Java getSAFeats()
     * Features are the DIFFERENCE between next state and current state features
     * @param {Array} obs - Current observation
     * @param {Array} act - Extended action (array of atomic actions)
     * @returns {Float64Array} State-change features (next - current)
     */
    getSAFeats(obs, act) {
        this.setStateFromObs(obs);

        // Apply all actions in the sequence
        for (let actI = 0; actI < act.length; actI++) {
            this.gameState.takeAction(act[actI]);
            this.gameState.update();
        }

        // Write current block to world and check for row clears
        this.gameState.writeCurrentBlock(this.gameState.worldState, true);
        this.gameState.checkIfRowAndScore();

        // Find features from difference in states
        return this.getSSFeats(obs, this.gameState.worldState);
    }

    /**
     * Get state-state features (difference between states) - EXACT from Java getSSFeats()
     * @param {Array} intStateVars - Current state observation
     * @param {Array} intNextStateVars - Next state (board after action)
     * @returns {Float64Array} Feature difference (next - current)
     */
    getSSFeats(intStateVars, intNextStateVars) {
        this._putSFeatsInArray(intStateVars, this.currSFeats);
        this._putSFeatsInArray(intNextStateVars, this.nextSFeats);

        const ssFeats = new Float64Array(TetrisFeatures.NUM_FEATS);
        for (let i = 0; i < TetrisFeatures.NUM_FEATS; i++) {
            ssFeats[i] = this.nextSFeats[i] - this.currSFeats[i];
        }
        return ssFeats;
    }

    /**
     * Get state-action features (for compatibility with non-extended action agents)
     * Uses action index to select slot in feature vector
     */
    getStateActionFeatures(obs, action) {
        const stateFeats = this.getStateFeatures(obs);

        // Create per-action feature vector (like RBFFeatures)
        const saFeats = new Float64Array(TetrisFeatures.NUM_FEATS * this.numActions);

        // Copy state features into the action's slot
        const offset = action * TetrisFeatures.NUM_FEATS;
        for (let i = 0; i < TetrisFeatures.NUM_FEATS; i++) {
            saFeats[offset + i] = stateFeats[i];
        }

        return saFeats;
    }

    /**
     * Get state features - EXACT from Java getSFeats()
     */
    getStateFeatures(obs) {
        const featsArray = new Float64Array(TetrisFeatures.NUM_FEATS);
        this._putSFeatsInArray(obs, featsArray);
        return featsArray;
    }

    /**
     * Compute features from board state - EXACT from Java putSFeatsInArray()
     */
    _putSFeatsInArray(intStateVars, featsArray) {
        // Initialize all to -1
        for (let i = 0; i < featsArray.length; i++) {
            featsArray[i] = -1;
        }

        featsArray[TetrisFeatures.NUM_HOLES_I] = 0;
        featsArray[TetrisFeatures.SUM_WELL_I] = 0;

        // Iterate through cells to compute column heights and holes
        for (let row = 0; row < this.worldHeight; row++) {
            for (let col = 0; col < this.worldWidth; col++) {
                const i = this._getIndex(row, col);
                if (intStateVars[i] > 0) {
                    // Filled cell
                    if (featsArray[TetrisFeatures.COL_HT_START_I + col] === -1) {
                        featsArray[TetrisFeatures.COL_HT_START_I + col] = row;
                    }
                    if (featsArray[TetrisFeatures.MAX_COL_HT_I] === -1) {
                        featsArray[TetrisFeatures.MAX_COL_HT_I] = row;
                    }
                } else {
                    // Empty cell
                    if (featsArray[TetrisFeatures.COL_HT_START_I + col] !== -1) {
                        featsArray[TetrisFeatures.NUM_HOLES_I] += 1;
                    }
                }
            }
        }

        // Set default values for columns with no blocks
        for (let col = 0; col < this.worldWidth; col++) {
            if (featsArray[TetrisFeatures.COL_HT_START_I + col] === -1) {
                featsArray[TetrisFeatures.COL_HT_START_I + col] = this.worldHeight;
            }
        }
        if (featsArray[TetrisFeatures.MAX_COL_HT_I] === -1) {
            featsArray[TetrisFeatures.MAX_COL_HT_I] = this.worldHeight;
        }

        // Compute column height differences
        for (let col = 0; col < this.worldWidth - 1; col++) {
            featsArray[TetrisFeatures.COL_DIFF_START_I + col] = Math.abs(
                featsArray[TetrisFeatures.COL_HT_START_I + col] -
                featsArray[TetrisFeatures.COL_HT_START_I + col + 1]
            );
        }

        // Compute well depths
        for (let col = 0; col < this.worldWidth; col++) {
            const wellDepth = this._getWellDepth(col, intStateVars);
            if (wellDepth > 0 &&
                (col < (this.worldWidth - 1) || TetrisFeatures.COUNT_LAST_COL_FOR_WELL_SUM)) {
                featsArray[TetrisFeatures.SUM_WELL_I] += wellDepth;
            }
            if (wellDepth > featsArray[TetrisFeatures.MAX_WELL_I]) {
                featsArray[TetrisFeatures.MAX_WELL_I] = wellDepth;
            }
        }

        // Compute squared features
        for (let i = 0; i < TetrisFeatures.SQUARED_FEATS_START_I; i++) {
            featsArray[TetrisFeatures.SQUARED_FEATS_START_I + i] = Math.pow(featsArray[i], 2.0);
            if (i <= TetrisFeatures.MAX_COL_HT_I || TetrisFeatures.SCALE_ALL_SQUARED_FEATS) {
                featsArray[TetrisFeatures.SQUARED_FEATS_START_I + i] /= TetrisFeatures.HT_SQ_SCALE;
            }
            if (TetrisFeatures.REMOVE_DECIMAL) {
                featsArray[TetrisFeatures.SQUARED_FEATS_START_I + i] =
                    Math.floor(featsArray[TetrisFeatures.SQUARED_FEATS_START_I + i]);
            }
        }
    }

    /**
     * Get linear index from row, col - EXACT from Java
     */
    _getIndex(row, col) {
        return (row * this.worldWidth) + col;
    }

    /**
     * Get well depth at column - EXACT from Java getWellDepth()
     */
    _getWellDepth(col, intStateVars) {
        let depth = 0;
        for (let row = 0; row < this.worldHeight; row++) {
            if (intStateVars[this._getIndex(row, col)] > 0) {
                // Encounter a filled space, stop counting
                break;
            } else {
                if (depth > 0) {
                    // If well-depth count has begun, don't require left and right to be filled
                    depth += 1;
                } else if (
                    (col === 0 || intStateVars[this._getIndex(row, col - 1)] > 0) &&
                    (col === this.worldWidth - 1 || intStateVars[this._getIndex(row, col + 1)] > 0)
                ) {
                    // Leftmost/rightmost column or adjacent cells are full - start count
                    depth += 1;
                }
            }
        }
        return depth;
    }

    /**
     * Set normalization bounds (for compatibility - not used for Tetris)
     */
    setNormBounds(min, max) {
        // Not applicable for Tetris features
    }

    /**
     * Set bias feature per action (for compatibility)
     */
    setBiasFeatPerAct(bias) {
        // Could add bias if needed
    }
}
