/**
 * Tetris - Tetris environment wrapper
 * EXACT port from Tetris.java in the RL-Glue Tetris environment
 *
 * Copyright 2007 Brian Tanner, Modified by Brad Knox
 * Licensed under the Apache License, Version 2.0
 */

import { TetrisState, TetrisActions } from './TetrisState.js';

export class Tetris {
    constructor() {
        this.gameState = new TetrisState();
        this.currentScore = 0;
        this.terminalScore = 0;  // EXACT from Java
    }

    /**
     * Get environment name
     */
    getName() {
        return 'Tetris';
    }

    /**
     * Initialize environment and return spec - EXACT from Java env_init()
     */
    init() {
        const boardSize = this.gameState.getHeight() * this.gameState.getWidth();
        const numPieces = this.gameState.possibleBlocks.length;

        return {
            // Observation is the board + 7 metadata values
            // Board cells are 0-1 (binary), plus block info
            obsRanges: this._buildObsRanges(),
            numActions: 5,  // FALL action has been removed in Java spec (0-4)
            discountFactor: 1.0,
            isEpisodic: true,
            rewardRange: [0, 8.0],
            extraInfo: {
                height: this.gameState.getHeight(),
                width: this.gameState.getWidth()
            }
        };
    }

    /**
     * Build observation ranges for feature generation
     */
    _buildObsRanges() {
        const ranges = [];
        const boardSize = this.gameState.getHeight() * this.gameState.getWidth();
        const numPieces = this.gameState.possibleBlocks.length;

        // Board cells (binary 0-1)
        for (let i = 0; i < boardSize; i++) {
            ranges.push([0, 1]);
        }

        // blockMobile (0 or 1)
        ranges.push([0, 1]);

        // currentBlockId (0 to numPieces-1)
        ranges.push([0, numPieces - 1]);

        // currentRotation (0 to 3)
        ranges.push([0, 3]);

        // currentX (0 to width-1)
        ranges.push([0, this.gameState.getWidth() - 1]);

        // currentY (0 to height-1)
        ranges.push([0, this.gameState.getHeight() - 1]);

        // worldWidth (constant)
        ranges.push([this.gameState.getWidth(), this.gameState.getWidth()]);

        // worldHeight (constant)
        ranges.push([this.gameState.getHeight(), this.gameState.getHeight()]);

        return ranges;
    }

    /**
     * Start a new episode - EXACT from Java env_start()
     */
    start() {
        this.gameState.reset();
        this.gameState.spawnBlock();
        this.gameState.blockMobile = true;
        this.currentScore = 0;

        return this.gameState.getObservation();
    }

    /**
     * Take a step - EXACT from Java env_step()
     * @param {number} action - Action to take (0-4, or 0-5 if FALL allowed)
     * @returns {{obs: number[], reward: number, terminal: boolean}}
     */
    step(action) {
        let theAction = action;

        if (theAction > 5 || theAction < 0) {
            console.error("Invalid action selected in Tetris: " + theAction);
            theAction = Math.floor(Math.random() * 5);
        }

        if (this.gameState.blockMobile) {
            this.gameState.takeAction(theAction);
            // Use updateWithPlacement for actual gameplay to write block and score
            this.gameState.updateWithPlacement();
        } else {
            this.gameState.spawnBlock();
        }

        const obs = this.gameState.getObservation();
        let reward;
        let terminal;

        if (!this.gameState.gameOver()) {
            terminal = false;
            reward = this.gameState.getScore() - this.currentScore;
            this.currentScore = this.gameState.getScore();
        } else {
            terminal = true;
            reward = this.terminalScore;
            this.currentScore = 0;
        }

        return {
            obs: obs,
            reward: reward,
            terminal: terminal
        };
    }

    /**
     * Get current state for visualization
     */
    getState() {
        return {
            worldState: this.gameState.getNumberedStateSnapShot(),
            worldWidth: this.gameState.getWidth(),
            worldHeight: this.gameState.getHeight(),
            currentBlockId: this.gameState.currentBlockId,
            currentRotation: this.gameState.currentRotation,
            currentX: this.gameState.currentX,
            currentY: this.gameState.currentY,
            blockMobile: this.gameState.blockMobile,
            score: this.gameState.getScore(),
            isGameOver: this.gameState.gameOver(),
            previousBlock: this.gameState.getPreviousBlock()
        };
    }

    /**
     * Get board dimensions
     */
    getDimensions() {
        return {
            width: this.gameState.getWidth(),
            height: this.gameState.getHeight()
        };
    }

    /**
     * Get piece colors for visualization - EXACT from Java TetrisBlocksComponent
     * Colors use makeColor(r, g, b) where values are 0-1 scaled to 0-255
     */
    static getPieceColors() {
        // EXACT from Java TetrisBlocksComponent.java (line 129-151)
        // Block IDs are 1-based when stored: 1=line, 2=square, 3=tri, 4=z, 5=s, 6=mirroredL, 7=L
        // But array is 0-indexed for colorIndex = cellValue - 1
        return [
            'rgb(128, 128, 128)',  // 1: Line (I) - gray (0.5, 0.5, 0.5)
            'rgb(128, 128, 128)',  // 2: Square (O) - gray (0.5, 0.5, 0.5)
            'rgb(128, 128, 128)',  // 3: Tri (T) - gray (0.5, 0.5, 0.5)
            'rgb(77, 13, 13)',     // 4: Z - dark red (0.3, 0.05, 0.05)
            'rgb(0, 77, 77)',      // 5: S - cyan (0.0, 0.3, 0.3)
            'rgb(0, 77, 77)',      // 6: mirrored L (J) - cyan (0.0, 0.3, 0.3)
            'rgb(77, 13, 13)'      // 7: L - dark red (0.3, 0.05, 0.05)
        ];
    }

    /**
     * Get action labels for display
     */
    getActionLabels() {
        return ['Left', 'Right', 'CW', 'CCW', 'None'];
    }

    /**
     * Get episode stats (for compatibility with TamerApp)
     */
    getEpisodeStats() {
        return {
            steps: 0,
            score: this.currentScore,
            gameOver: this.gameState.gameOver()
        };
    }

    /**
     * Render the environment - EXACT colors from Java TetrisBlocksComponent
     */
    render(ctx, canvasWidth, canvasHeight) {
        const state = this.getState();
        const { worldWidth, worldHeight, worldState, currentBlockId, previousBlock } = state;

        // Calculate cell size to fit canvas
        const cellWidth = Math.floor((canvasWidth - 100) / worldWidth);
        const cellHeight = Math.floor((canvasHeight - 50) / worldHeight);
        const cellSize = Math.min(cellWidth, cellHeight);

        const offsetX = (canvasWidth - cellSize * worldWidth) / 2;
        const offsetY = 10;

        // Draw border - EXACT from Java (Color.GRAY)
        ctx.fillStyle = 'rgb(128, 128, 128)';
        ctx.fillRect(offsetX - 2, offsetY - 2, cellSize * worldWidth + 4, cellSize * worldHeight + 4);

        // Draw white background - EXACT from Java (Color.WHITE)
        ctx.fillStyle = 'rgb(255, 255, 255)';
        ctx.fillRect(offsetX, offsetY, cellSize * worldWidth, cellSize * worldHeight);

        // Get piece colors
        const colors = Tetris.getPieceColors();

        // Draw blocks
        for (let y = 0; y < worldHeight; y++) {
            for (let x = 0; x < worldWidth; x++) {
                const cellValue = worldState[y * worldWidth + x];
                const linearIndex = y * worldWidth + x;

                if (cellValue > 0) {
                    // Check if this is part of the previous block (most recently placed)
                    let isPreviousBlock = false;
                    if (previousBlock) {
                        for (let i = 0; i < previousBlock.length; i++) {
                            if (previousBlock[i] === linearIndex) {
                                isPreviousBlock = true;
                                break;
                            }
                        }
                    }

                    // Previous block is rendered in BLACK - EXACT from Java
                    if (isPreviousBlock) {
                        ctx.fillStyle = 'rgb(0, 0, 0)';
                    } else {
                        const colorIndex = (cellValue - 1) % colors.length;
                        ctx.fillStyle = colors[colorIndex];
                    }

                    // Use fill3DRect-like effect
                    ctx.fillRect(
                        offsetX + x * cellSize + 1,
                        offsetY + y * cellSize + 1,
                        cellSize - 2,
                        cellSize - 2
                    );

                    // Add 3D highlight effect
                    ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
                    ctx.fillRect(
                        offsetX + x * cellSize + 1,
                        offsetY + y * cellSize + 1,
                        cellSize - 2,
                        2
                    );
                    ctx.fillRect(
                        offsetX + x * cellSize + 1,
                        offsetY + y * cellSize + 1,
                        2,
                        cellSize - 2
                    );
                }
            }
        }

        // Draw border - EXACT from Java (g.drawRect with Color.GRAY)
        ctx.strokeStyle = 'rgb(128, 128, 128)';
        ctx.lineWidth = 1;
        ctx.strokeRect(offsetX, offsetY, cellSize * worldWidth, cellSize * worldHeight);

        // Draw score
        ctx.fillStyle = '#000000';
        ctx.font = '16px Arial';
        ctx.textAlign = 'left';
        ctx.fillText(`Lines: ${state.score}`, offsetX, offsetY + worldHeight * cellSize + 25);

        // Draw game over message
        if (state.isGameOver) {
            ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
            ctx.fillRect(offsetX, offsetY + worldHeight * cellSize / 2 - 30, worldWidth * cellSize, 60);
            ctx.fillStyle = '#ff0000';
            ctx.font = 'bold 24px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('GAME OVER', offsetX + worldWidth * cellSize / 2, offsetY + worldHeight * cellSize / 2 + 8);
        }
    }
}

// Re-export TetrisActions for convenience
export { TetrisActions };
