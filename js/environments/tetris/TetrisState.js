/**
 * TetrisState - Tetris game state management
 * EXACT port from TetrisState.java in the RL-Glue Tetris environment
 *
 * Copyright 2007 Brian Tanner, Modified by Brad Knox
 * Licensed under the Apache License, Version 2.0
 */

import { TetrisPiece } from './TetrisPiece.js';

// Action values - EXACT from Java
export const TetrisActions = {
    LEFT: 0,   // Action value for a move left
    RIGHT: 1,  // Action value for a move right
    CW: 2,     // Action value for a clockwise rotation
    CCW: 3,    // Action value for a counter clockwise rotation
    NONE: 4,   // The no-action Action
    FALL: 5    // Fall down
};

export const NUM_BLOCK_TYPES = 7;

export class TetrisState {
    constructor(stateToCopy = null) {
        if (stateToCopy) {
            // Copy constructor - EXACT from Java
            this.blockMobile = stateToCopy.blockMobile;
            this.currentBlockId = stateToCopy.currentBlockId;
            this.currentRotation = stateToCopy.currentRotation;
            this.currentX = stateToCopy.currentX;
            this.currentY = stateToCopy.currentY;
            this.score = stateToCopy.score;
            this.is_game_over = stateToCopy.is_game_over;
            this.worldWidth = stateToCopy.worldWidth;
            this.worldHeight = stateToCopy.worldHeight;

            this.worldState = new Int32Array(stateToCopy.worldState.length);
            for (let i = 0; i < this.worldState.length; i++) {
                this.worldState[i] = stateToCopy.worldState[i];
            }

            // Copy possible blocks (reference copy is fine since pieces don't change)
            this.possibleBlocks = stateToCopy.possibleBlocks.slice();

            // Copy previous block info
            this.previousBlock = stateToCopy.previousBlock ? stateToCopy.previousBlock.slice() : null;
            this.secToLastBlock = stateToCopy.secToLastBlock ? stateToCopy.secToLastBlock.slice() : null;
        } else {
            // Default constructor - EXACT from Java
            this.worldWidth = 10;
            this.worldHeight = 20;

            this.possibleBlocks = [
                TetrisPiece.makeLine(),
                TetrisPiece.makeSquare(),
                TetrisPiece.makeTri(),
                TetrisPiece.makeSShape(),
                TetrisPiece.makeZShape(),
                TetrisPiece.makeLShape(),
                TetrisPiece.makeJShape()
            ];

            this.worldState = new Int32Array(this.worldHeight * this.worldWidth);

            this.previousBlock = null;
            this.secToLastBlock = null;

            this.reset();
        }
    }

    /**
     * Reset game state - EXACT from Java
     */
    reset() {
        this.currentX = Math.floor(this.worldWidth / 2) - 1;
        this.currentY = 0;
        this.score = 0;
        for (let i = 0; i < this.worldState.length; i++) {
            this.worldState[i] = 0;
        }
        this.currentRotation = 0;
        this.is_game_over = false;
        this.blockMobile = true;
        this.currentBlockId = 0;
    }

    /**
     * Get observation - EXACT from Java
     * Returns array with worldState + 7 additional values
     */
    getObservation() {
        const o = new Array(this.worldState.length + 7);

        for (let i = 0; i < this.worldState.length; i++) {
            o[i] = this.worldState[i];
        }

        o[this.worldState.length] = this.blockMobile ? 1 : 0;
        o[this.worldState.length + 1] = this.currentBlockId;
        o[this.worldState.length + 2] = this.currentRotation;
        o[this.worldState.length + 3] = this.currentX;
        o[this.worldState.length + 4] = this.currentY;
        o[this.worldState.length + 5] = this.worldWidth;
        o[this.worldState.length + 6] = this.worldHeight;

        return o;
    }

    /**
     * Write current block to game world - EXACT from Java writeCurrentBlock()
     */
    writeCurrentBlock(gameWorld, onSomething) {
        if (onSomething) {
            this.secToLastBlock = this.previousBlock;
            this.previousBlock = new Array(4);
        }

        const thisPiece = this.possibleBlocks[this.currentBlockId].getShape(this.currentRotation);
        let cellIndex = 0;

        for (let y = 0; y < thisPiece[0].length; ++y) {
            for (let x = 0; x < thisPiece.length; ++x) {
                if (thisPiece[x][y] !== 0) {
                    const linearIndex = this._calculateLinearArrayPosition(this.currentX + x, this.currentY + y);
                    if (linearIndex < 0) {
                        console.error(`Bogus linear index ${linearIndex} for ${this.currentX} + ${x}, ${this.currentY} + ${y}`);
                        return;
                    }

                    const hollowAddition = 0; // onSomething ? NUM_BLOCK_TYPES : 0;

                    // Save previous block if onSomething
                    if (onSomething) {
                        this.previousBlock[cellIndex] = linearIndex;
                        cellIndex++;
                    }

                    gameWorld[linearIndex] = this.currentBlockId + 1 + hollowAddition;
                }
            }
        }
    }

    /**
     * Remove hollowness from previous block - EXACT from Java
     */
    _removeHollowness(gameWorld) {
        if (this.secToLastBlock !== null) {
            for (let i = 0; i < this.secToLastBlock.length; i++) {
                const linearIndex = this.secToLastBlock[i];
                gameWorld[linearIndex] = gameWorld[linearIndex] - NUM_BLOCK_TYPES;
            }
        }
    }

    /**
     * Get previous block
     */
    getPreviousBlock() {
        return this.previousBlock;
    }

    /**
     * Get second to last block
     */
    getSecToLastBlock() {
        return this.secToLastBlock;
    }

    /**
     * Check if game is over - EXACT from Java
     */
    gameOver() {
        return this.is_game_over;
    }

    /**
     * Take action (apply move without default fall) - EXACT from Java take_action()
     * @param {number} theAction - Action to take
     * @returns {boolean} true if the move was legal, false otherwise
     */
    takeAction(theAction) {
        if (theAction > 5 || theAction < 0) {
            console.error("Invalid action selected in Tetris: " + theAction);
            theAction = Math.floor(Math.random() * 6);
        }

        let nextRotation = this.currentRotation;
        let nextX = this.currentX;
        let nextY = this.currentY;

        switch (theAction) {
            case TetrisActions.CW:
                nextRotation = (this.currentRotation + 1) % 4;
                break;
            case TetrisActions.CCW:
                nextRotation = (this.currentRotation - 1);
                if (nextRotation < 0) {
                    nextRotation = 3;
                }
                break;
            case TetrisActions.LEFT:
                nextX = this.currentX - 1;
                break;
            case TetrisActions.RIGHT:
                nextX = this.currentX + 1;
                break;
            case TetrisActions.FALL:
                nextY = this.currentY;

                let isInBounds = true;
                let isColliding = false;

                // Fall until you hit something then back up once
                while (isInBounds && !isColliding) {
                    nextY++;
                    isInBounds = this._inBounds(nextX, nextY, nextRotation);
                    if (isInBounds) {
                        isColliding = this._colliding(nextX, nextY, nextRotation);
                    }
                }
                nextY--;
                break;
            default:
                break;
        }

        let legalMove = false;
        // Check if the resulting position is legal. If so, accept it.
        // Otherwise, don't change anything
        if (this._inBounds(nextX, nextY, nextRotation)) {
            if (!this._colliding(nextX, nextY, nextRotation)) {
                this.currentRotation = nextRotation;
                this.currentX = nextX;
                this.currentY = nextY;
                legalMove = true;
            }
        }
        return legalMove;
    }

    /**
     * Check if rotation is equivalent - EXACT from Java equivRotation()
     * For certain pieces, some rotations produce identical shapes
     * @param {number} rotation - Rotation to compare against current
     * @returns {boolean} true if rotations are equivalent
     */
    equivRotation(rotation) {
        if (this.currentRotation === rotation) {
            return true;
        }
        // Line (0), S-shape (3), Z-shape (4) have 2-fold symmetry
        if (this.currentBlockId === 0 ||
            this.currentBlockId === 3 ||
            this.currentBlockId === 4) {
            // Both are odd or both are even
            if ((this.currentRotation + rotation) % 2 === 0) {
                return true;
            }
        }
        // Square (1) has 4-fold symmetry - all rotations are equivalent
        if (this.currentBlockId === 1) {
            return true;
        }
        return false;
    }

    /**
     * Calculate linear array position - EXACT from Java
     */
    _calculateLinearArrayPosition(x, y) {
        const returnValue = y * this.worldWidth + x;
        return returnValue;
    }

    /**
     * Check if piece is colliding - EXACT from Java
     */
    _colliding(checkX, checkY, checkOrientation) {
        const thePiece = this.possibleBlocks[this.currentBlockId].getShape(checkOrientation);

        for (let y = 0; y < thePiece[0].length; ++y) {
            for (let x = 0; x < thePiece.length; ++x) {
                if (thePiece[x][y] !== 0) {
                    // Check if out of bounds (colliding with wall)
                    if (checkY + y < 0 || checkX + x < 0) {
                        return true;
                    }
                    if (checkY + y >= this.worldHeight || checkX + x >= this.worldWidth) {
                        return true;
                    }

                    // Check if hits another piece
                    const linearArrayIndex = this._calculateLinearArrayPosition(checkX + x, checkY + y);
                    if (this.worldState[linearArrayIndex] !== 0) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    /**
     * Check collision only for in-bounds spots - EXACT from Java
     */
    _collidingCheckOnlySpotsInBounds(checkX, checkY, checkOrientation) {
        const thePiece = this.possibleBlocks[this.currentBlockId].getShape(checkOrientation);

        for (let y = 0; y < thePiece[0].length; ++y) {
            for (let x = 0; x < thePiece.length; ++x) {
                if (thePiece[x][y] !== 0) {
                    // This checks to see if x and y are in bounds
                    if ((checkX + x >= 0 && checkX + x < this.worldWidth &&
                         checkY + y >= 0 && checkY + y < this.worldHeight)) {
                        // This array location is in bounds
                        // Check if it hits another piece
                        const linearArrayIndex = this._calculateLinearArrayPosition(checkX + x, checkY + y);
                        if (this.worldState[linearArrayIndex] !== 0) {
                            return true;
                        }
                    }
                }
            }
        }
        return false;
    }

    /**
     * Check if piece is in bounds - EXACT from Java
     */
    _inBounds(checkX, checkY, checkOrientation) {
        const thePiece = this.possibleBlocks[this.currentBlockId].getShape(checkOrientation);

        for (let y = 0; y < thePiece[0].length; ++y) {
            for (let x = 0; x < thePiece.length; ++x) {
                if (thePiece[x][y] !== 0) {
                    if (!(checkX + x >= 0 && checkX + x < this.worldWidth &&
                          checkY + y >= 0 && checkY + y < this.worldHeight)) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    /**
     * Check if next position is in bounds
     */
    _nextInBounds() {
        return this._inBounds(this.currentX, this.currentY + 1, this.currentRotation);
    }

    /**
     * Check if next position is colliding
     */
    _nextColliding() {
        return this._colliding(this.currentX, this.currentY + 1, this.currentRotation);
    }

    /**
     * Update game state after action - EXACT from Java TAMER TetrisState.update()
     * NOTE: In the TAMER version, writeCurrentBlock and checkIfRowAndScore are
     * NOT called here - they are commented out. This is important for tree search
     * during extended action generation to work correctly.
     */
    update() {
        // Sanity check
        if (!this._inBounds(this.currentX, this.currentY, this.currentRotation)) {
            console.error("In TetrisState the Current Position of the board is Out Of Bounds... Consistency Check Failed");
        }

        // onSomething means we're done with this piece
        let onSomething = false;
        if (!this._nextInBounds()) {
            onSomething = true;
        }
        if (!onSomething) {
            if (this._nextColliding()) {
                onSomething = true;
            }
        }

        if (onSomething) {
            this.blockMobile = false;
            // NOTE: These are commented out in Java TAMER TetrisState.update()
            // They are called explicitly where needed (e.g., in getSAFeats)
            // this.writeCurrentBlock(this.worldState, true);
            // this.checkIfRowAndScore();
        } else {
            // fall
            this.currentY += 1;
        }
    }

    /**
     * Update game state after action - version that writes block and scores
     * Used by the environment for actual gameplay (not tree search)
     */
    updateWithPlacement() {
        // Sanity check
        if (!this._inBounds(this.currentX, this.currentY, this.currentRotation)) {
            console.error("In TetrisState the Current Position of the board is Out Of Bounds... Consistency Check Failed");
        }

        // onSomething means we're done with this piece
        let onSomething = false;
        if (!this._nextInBounds()) {
            onSomething = true;
        }
        if (!onSomething) {
            if (this._nextColliding()) {
                onSomething = true;
            }
        }

        if (onSomething) {
            this.blockMobile = false;
            this.writeCurrentBlock(this.worldState, true);
            this.checkIfRowAndScore();
        } else {
            // fall
            this.currentY += 1;
        }
    }

    /**
     * Spawn a new block - EXACT from Java
     */
    spawnBlock() {
        this.blockMobile = true;

        this.currentBlockId = Math.floor(Math.random() * this.possibleBlocks.length);
        this.currentRotation = 0;
        this.currentX = Math.floor(this.worldWidth / 2) - 2;
        this.currentY = -4;

        let hitOnWayIn = false;
        while (!this._inBounds(this.currentX, this.currentY, this.currentRotation)) {
            hitOnWayIn = this._collidingCheckOnlySpotsInBounds(this.currentX, this.currentY, this.currentRotation);
            this.currentY++;
        }

        this.is_game_over = this._colliding(this.currentX, this.currentY, this.currentRotation) || hitOnWayIn;
        if (this.is_game_over) {
            this.previousBlock = null;
            this.secToLastBlock = null;
            this.blockMobile = false;
        }
    }

    /**
     * Check for completed rows and score - EXACT from Java checkIfRowAndScore()
     */
    checkIfRowAndScore() {
        let numRowsCleared = 0;

        // Start at the bottom, work way up
        for (let y = this.worldHeight - 1; y >= 0; --y) {
            if (this._isRow(y)) {
                this._removeRow(y);
                numRowsCleared += 1;
                y += 1;  // Check this row again
            }
        }

        // Scoring: 1 line = 1, 2 lines = 2, 3 lines = 4, 4 lines = 8
        // Math.pow(2.0, numRowsCleared-1)
        this.score += Math.pow(2.0, numRowsCleared - 1);
    }

    /**
     * Check if row is complete - EXACT from Java
     */
    _isRow(y) {
        for (let x = 0; x < this.worldWidth; ++x) {
            const linearIndex = this._calculateLinearArrayPosition(x, y);
            if (this.worldState[linearIndex] === 0) {
                return false;
            }
        }
        return true;
    }

    /**
     * Remove completed row - EXACT from Java
     */
    _removeRow(y) {
        if (!this._isRow(y)) {
            console.error("In TetrisState removeRow you have tried to remove a row which is not complete");
            return;
        }

        // Clear the row
        for (let x = 0; x < this.worldWidth; ++x) {
            const linearIndex = this._calculateLinearArrayPosition(x, y);
            this.worldState[linearIndex] = 0;
        }

        // Copy each row down one (except the top)
        for (let ty = y; ty > 0; --ty) {
            for (let x = 0; x < this.worldWidth; ++x) {
                const linearIndexTarget = this._calculateLinearArrayPosition(x, ty);
                const linearIndexSource = this._calculateLinearArrayPosition(x, ty - 1);
                this.worldState[linearIndexTarget] = this.worldState[linearIndexSource];
            }
        }

        // Clear the top row
        for (let x = 0; x < this.worldWidth; ++x) {
            const linearIndex = this._calculateLinearArrayPosition(x, 0);
            this.worldState[linearIndex] = 0;
        }
    }

    /**
     * Get current score - EXACT from Java
     */
    getScore() {
        return this.score;
    }

    /**
     * Get world width
     */
    getWidth() {
        return this.worldWidth;
    }

    /**
     * Get world height
     */
    getHeight() {
        return this.worldHeight;
    }

    /**
     * Get numbered state snapshot (includes current block) - EXACT from Java
     */
    getNumberedStateSnapShot() {
        const numberedStateCopy = new Int32Array(this.worldState.length);
        for (let i = 0; i < this.worldState.length; i++) {
            numberedStateCopy[i] = this.worldState[i];
        }
        this.writeCurrentBlock(numberedStateCopy, false);
        return numberedStateCopy;
    }

    /**
     * Get current piece ID
     */
    getCurrentPiece() {
        return this.currentBlockId;
    }

    /**
     * Print state for debugging
     */
    printState() {
        for (let i = 0; i < this.worldHeight; i++) {
            let row = '';
            for (let j = 0; j < this.worldWidth; j++) {
                row += this.worldState[i * this.worldWidth + j];
            }
            console.log(row);
        }
        console.log("-------------");
    }
}
