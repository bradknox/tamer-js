/**
 * LoopMaze Environment
 * EXACT port from LoopMaze.java and LoopMazeState.java in the TAMER project
 *
 * A gridworld maze from the TAMER paper.
 */
import { Environment } from './Environment.js';

export class LoopMaze extends Environment {
    constructor(options = {}) {
        super();

        // State map - EXACT from Java LoopMazeState.java
        // 1 = traversable, 0 = wall, 2 = secret state
        this.stateMap = [
            [1, 1, 1, 1, 1, 1],  // Row 0 (y=0)
            [1, 0, 1, 1, 1, 1],  // Row 1
            [1, 0, 1, 0, 0, 1],  // Row 2
            [1, 2, 1, 1, 1, 1],  // Row 3
            [1, 0, 1, 1, 1, 1],  // Row 4
            [1, 1, 1, 1, 1, 1]   // Row 5 (y=5)
        ];

        // Vertical walls (control horizontal movement) - EXACT from Java
        // vertWalls[y][x] = 1 means wall between (x,y) and (x+1,y)
        this.vertWalls = [
            [0, 0, 0, 0, 1],  // y=0
            [1, 1, 0, 0, 0],  // y=1
            [1, 1, 1, 0, 1],  // y=2
            [2, 2, 0, 0, 0],  // y=3 (2 = secret wall)
            [1, 1, 0, 0, 0],  // y=4
            [0, 0, 0, 0, 0]   // y=5
        ];

        // Horizontal walls (control vertical movement) - EXACT from Java
        // horWalls[y][x] = 1 means wall between (x,y) and (x,y+1)
        this.horWalls = [
            [0, 1, 1, 1, 1, 0],  // between y=0 and y=1
            [0, 0, 0, 1, 1, 0],  // between y=1 and y=2
            [0, 0, 0, 1, 1, 0],  // between y=2 and y=3
            [0, 0, 0, 0, 0, 0],  // between y=3 and y=4
            [0, 1, 1, 1, 1, 0]   // between y=4 and y=5
        ];

        // World dimensions - EXACT from Java
        this.worldDims = [this.stateMap[0].length, this.stateMap.length];  // [width, height] = [6, 6]

        // Start position - EXACT from Java defaultInitPosition = {4, 0}
        this.startX = options.startX ?? 4;
        this.startY = options.startY ?? 0;

        // Goal location - EXACT from Java goalLoc = {5, 0}
        this.goalX = 5;
        this.goalY = 0;

        // Allow secret paths option
        this.allowSecretPaths = options.allowSecretPaths ?? false;

        // Agent position
        this.agentX = 0;
        this.agentY = 0;

        // Random start option
        this.randomStartStates = options.randomStartStates ?? false;
        this.transitionNoise = options.transitionNoise ?? 0.0;

        // Actions: 0=right, 1=left, 2=down, 3=up - EXACT from Java
        this.numActions = 4;

        // Observation ranges (x and y coordinates)
        this.obsRanges = [
            [0, this.worldDims[0] - 1],
            [0, this.worldDims[1] - 1]
        ];

        // Reward settings - EXACT from Java
        this.rewardPerStep = -1.0;
        this.rewardAtGoal = 0.0;
        this.rewardAtFail = -20.0;

        // Maximum steps per episode
        this.maxSteps = options.maxSteps ?? 200;

        // Track last action for visualization
        this.lastAction = null;
        this.lastAgentX = null;
        this.lastAgentY = null;
        this.hasMoved = true;
    }

    /**
     * Check if state is legal - EXACT from Java isStateLegal()
     */
    _isStateLegal(x, y) {
        if (x < 0 || x >= this.worldDims[0] || y < 0 || y >= this.worldDims[1]) {
            return false;
        }
        const stateVal = this.stateMap[y][x];
        if (stateVal === 0) return false;
        if (stateVal === 2 && !this.allowSecretPaths) return false;
        return true;
    }

    /**
     * Check if move is legal - EXACT from Java isMoveLegal()
     */
    _isMoveLegal(fromX, fromY, toX, toY) {
        // Check destination is legal
        if (!this._isStateLegal(toX, toY)) return false;

        // Check walls
        if (toX > fromX) {
            // Moving right - check vertical wall
            if (this.vertWalls[fromY][fromX] === 1) return false;
            if (this.vertWalls[fromY][fromX] === 2 && !this.allowSecretPaths) return false;
        } else if (toX < fromX) {
            // Moving left - check vertical wall
            if (this.vertWalls[fromY][toX] === 1) return false;
            if (this.vertWalls[fromY][toX] === 2 && !this.allowSecretPaths) return false;
        } else if (toY > fromY) {
            // Moving down - check horizontal wall
            if (this.horWalls[fromY][fromX] === 1) return false;
            if (this.horWalls[fromY][fromX] === 2 && !this.allowSecretPaths) return false;
        } else if (toY < fromY) {
            // Moving up - check horizontal wall
            if (this.horWalls[toY][fromX] === 1) return false;
            if (this.horWalls[toY][fromX] === 2 && !this.allowSecretPaths) return false;
        }

        return true;
    }

    /**
     * Check if in goal region - EXACT from Java inGoalRegion()
     */
    _inGoalRegion(x, y) {
        return x === this.goalX && y === this.goalY;
    }

    /**
     * Check if in fail region - EXACT from Java inFailRegion()
     * Note: Java has this as null/empty by default
     */
    _inFailRegion(x, y) {
        return false;  // No fail region in default Java config
    }

    /**
     * Get reward - EXACT from Java getReward()
     */
    _getReward(x, y) {
        if (this._inGoalRegion(x, y)) {
            return this.rewardAtGoal;
        }
        if (this._inFailRegion(x, y)) {
            return this.rewardAtFail;
        }
        return this.rewardPerStep;
    }

    /**
     * Initialize the environment
     */
    init() {
        return {
            numActions: this.numActions,
            obsRanges: this.obsRanges,
            actionLabels: ['Right', 'Left', 'Down', 'Up']  // EXACT from Java action order
        };
    }

    /**
     * Start a new episode - EXACT from Java reset()/env_start()
     */
    start() {
        if (this.randomStartStates) {
            // Random start from legal positions - EXACT from Java sampleEnvStart lines 236-243
            do {
                this.agentX = Math.floor(Math.random() * this.worldDims[0]);
                this.agentY = Math.floor(Math.random() * this.worldDims[1]);
            } while (this._inGoalRegion(this.agentX, this.agentY) ||
                     this._inFailRegion(this.agentX, this.agentY) ||
                     !this._isStateLegal(this.agentX, this.agentY));
        } else {
            this.agentX = this.startX;
            this.agentY = this.startY;
        }

        this.episodeSteps = 0;
        this.episodeReward = 0;
        this.isTerminal = false;
        this.currentObs = [this.agentX, this.agentY];
        this.lastAction = null;
        this.lastAgentX = null;
        this.lastAgentY = null;
        this.hasMoved = true;
        return this.currentObs;
    }

    /**
     * Take a step - EXACT from Java update()/env_step()
     */
    step(action) {
        if (this.isTerminal) {
            return { obs: this.currentObs, reward: 0, terminal: true };
        }

        // Store previous position
        this.lastAgentX = this.agentX;
        this.lastAgentY = this.agentY;
        this.lastAction = action;

        // Apply transition noise
        if (Math.random() < this.transitionNoise) {
            action = Math.floor(Math.random() * this.numActions);
        }

        // Calculate new position - EXACT from Java action mapping
        // 0=right, 1=left, 2=down, 3=up
        let newX = this.agentX;
        let newY = this.agentY;

        switch (action) {
            case 0: newX += 1; break;  // Right
            case 1: newX -= 1; break;  // Left
            case 2: newY += 1; break;  // Down
            case 3: newY -= 1; break;  // Up
        }

        // Check if move is legal
        if (this._isMoveLegal(this.agentX, this.agentY, newX, newY)) {
            this.agentX = newX;
            this.agentY = newY;
            this.hasMoved = true;
        } else {
            this.hasMoved = false;
        }

        // Update observation
        this.currentObs = [this.agentX, this.agentY];

        // Get reward
        const reward = this._getReward(this.agentX, this.agentY);

        // Check terminal conditions
        if (this._inGoalRegion(this.agentX, this.agentY) ||
            this._inFailRegion(this.agentX, this.agentY)) {
            this.isTerminal = true;
        }

        // Update statistics
        this.episodeSteps++;
        this.totalSteps++;
        this.episodeReward += reward;

        // Check max steps
        if (this.episodeSteps >= this.maxSteps) {
            this.isTerminal = true;
        }

        return {
            obs: this.currentObs,
            reward: reward,
            terminal: this.isTerminal
        };
    }

    /**
     * Render the maze
     */
    render(ctx, width, height) {
        const cellWidth = width / this.worldDims[0];
        const cellHeight = height / this.worldDims[1];

        // Draw maze cells - EXACT from Java MazeMapComponent.java
        for (let y = 0; y < this.worldDims[1]; y++) {
            for (let x = 0; x < this.worldDims[0]; x++) {
                const cell = this.stateMap[y][x];
                const px = x * cellWidth;
                const py = y * cellHeight;

                // Fill based on cell type
                // Java line 143: if (stateMap[y][x] != 1) -> LIGHT_GRAY
                if (cell !== 1) {
                    ctx.fillStyle = '#C0C0C0';  // LIGHT_GRAY for non-traversable (0 and 2)
                } else {
                    ctx.fillStyle = '#FFFFFF';  // White for traversable
                }
                ctx.fillRect(px, py, cellWidth, cellHeight);

                // Draw grid lines - EXACT from Java lines 125-135
                ctx.strokeStyle = '#808080';  // GRAY
                ctx.lineWidth = 1;
                ctx.strokeRect(px, py, cellWidth, cellHeight);
            }
        }

        // Draw walls (thicker lines)
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 3;

        // Draw vertical walls - EXACT from Java: draw if value != 0
        // Secret walls (value 2) are drawn visually but can be traversed
        for (let y = 0; y < this.worldDims[1]; y++) {
            for (let x = 0; x < this.worldDims[0] - 1; x++) {
                if (this.vertWalls[y][x] !== 0) {
                    const px = (x + 1) * cellWidth;
                    const py = y * cellHeight;
                    ctx.beginPath();
                    ctx.moveTo(px, py);
                    ctx.lineTo(px, py + cellHeight);
                    ctx.stroke();
                }
            }
        }

        // Draw horizontal walls - EXACT from Java: draw if value != 0
        for (let y = 0; y < this.worldDims[1] - 1; y++) {
            for (let x = 0; x < this.worldDims[0]; x++) {
                if (this.horWalls[y][x] !== 0) {
                    const px = x * cellWidth;
                    const py = (y + 1) * cellHeight;
                    ctx.beginPath();
                    ctx.moveTo(px, py);
                    ctx.lineTo(px + cellWidth, py);
                    ctx.stroke();
                }
            }
        }

        // Agent dimensions
        const agentWidth = cellWidth * 0.5;
        const agentHeight = cellHeight * 0.5;

        // Agent position (centered in cell)
        const agentPx = this.agentX * cellWidth + cellWidth * 0.25;
        const agentPy = this.agentY * cellHeight + cellHeight * 0.25;

        // Draw agent rectangle - olive green
        ctx.fillStyle = 'rgb(70, 100, 20)';
        ctx.fillRect(agentPx, agentPy, agentWidth, agentHeight);
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 2;
        ctx.strokeRect(agentPx, agentPy, agentWidth, agentHeight);

        // Eye parameters - EXACT from Java MazeMapComponent.java lines 276-280
        const eyeShiftVal = 0.22;
        const eyeDistFromCenter = 0.1;
        const eyeHt = 0.12;  // Java: eyeHt = 0.12
        const eyeWd = 0.12;  // Java: eyeWidth = 0.12
        const eyeWidth = cellWidth * eyeWd;
        const eyeHeight = cellHeight * eyeHt;
        const eyeLineScale = 1.3;  // Java uses 1.3

        // Agent center
        const agentCenterX = agentPx + agentWidth / 2;
        const agentCenterY = agentPy + agentHeight / 2;

        // Determine eye position based on last action
        // Java: 0=right, 1=left, 2=down, 3=up
        let eyeShiftX = 0;
        let eyeShiftY = 0;
        let horizontalAct = false;

        if (this.lastAction !== null) {
            switch (this.lastAction) {
                case 0:  // Right
                    eyeShiftX = cellWidth * eyeShiftVal;
                    horizontalAct = true;
                    break;
                case 1:  // Left
                    eyeShiftX = -cellWidth * eyeShiftVal;
                    horizontalAct = true;
                    break;
                case 2:  // Down
                    eyeShiftY = cellHeight * eyeShiftVal;
                    horizontalAct = false;
                    break;
                case 3:  // Up
                    eyeShiftY = -cellHeight * eyeShiftVal;
                    horizontalAct = false;
                    break;
            }
        }

        // Calculate eye positions
        let eye1X, eye1Y, eye2X, eye2Y;

        if (horizontalAct) {
            eye1X = agentCenterX + eyeShiftX - eyeWidth / 2;
            eye1Y = agentCenterY + cellHeight * eyeDistFromCenter - eyeHeight / 2;
            eye2X = agentCenterX + eyeShiftX - eyeWidth / 2;
            eye2Y = agentCenterY - cellHeight * eyeDistFromCenter - eyeHeight / 2;
        } else {
            eye1X = agentCenterX + cellWidth * eyeDistFromCenter + eyeShiftX - eyeWidth / 2;
            eye1Y = agentCenterY + eyeShiftY - eyeHeight / 2;
            eye2X = agentCenterX - cellWidth * eyeDistFromCenter + eyeShiftX - eyeWidth / 2;
            eye2Y = agentCenterY + eyeShiftY - eyeHeight / 2;
        }

        // Draw eyes - black outline
        ctx.fillStyle = '#000000';
        ctx.beginPath();
        ctx.ellipse(eye1X + eyeWidth / 2, eye1Y + eyeHeight / 2,
                    eyeWidth * eyeLineScale / 2, eyeHeight * eyeLineScale / 2, 0, 0, Math.PI * 2);
        ctx.fill();
        ctx.beginPath();
        ctx.ellipse(eye2X + eyeWidth / 2, eye2Y + eyeHeight / 2,
                    eyeWidth * eyeLineScale / 2, eyeHeight * eyeLineScale / 2, 0, 0, Math.PI * 2);
        ctx.fill();

        // Draw eyes - white fill
        ctx.fillStyle = '#FFFFFF';
        ctx.beginPath();
        ctx.ellipse(eye1X + eyeWidth / 2, eye1Y + eyeHeight / 2,
                    eyeWidth / 2, eyeHeight / 2, 0, 0, Math.PI * 2);
        ctx.fill();
        ctx.beginPath();
        ctx.ellipse(eye2X + eyeWidth / 2, eye2Y + eyeHeight / 2,
                    eyeWidth / 2, eyeHeight / 2, 0, 0, Math.PI * 2);
        ctx.fill();

        // Draw collision X mark if agent didn't move
        if (!this.hasMoved) {
            ctx.strokeStyle = '#FF0000';
            ctx.lineWidth = 2;

            let xCenterX = agentCenterX + eyeShiftX * 1.5;
            let xCenterY = agentCenterY + eyeShiftY * 1.5;
            const xSize = cellWidth * 0.15;

            ctx.beginPath();
            ctx.moveTo(xCenterX - xSize, xCenterY - xSize);
            ctx.lineTo(xCenterX + xSize, xCenterY + xSize);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(xCenterX + xSize, xCenterY - xSize);
            ctx.lineTo(xCenterX - xSize, xCenterY + xSize);
            ctx.stroke();
        }

        // Draw goal AFTER agent - EXACT from Java MazeMapComponent.java lines 382-389
        // Java draws goal with alpha 0.7 on top of agent
        ctx.globalAlpha = 0.7;
        ctx.fillStyle = 'rgb(107, 157, 174)';
        ctx.fillRect(this.goalX * cellWidth, this.goalY * cellHeight, cellWidth, cellHeight);
        ctx.globalAlpha = 1.0;
    }

    /**
     * Get action labels - EXACT from Java action order
     */
    getActionLabels() {
        return ['Right', 'Left', 'Down', 'Up'];
    }

    /**
     * Get environment name
     */
    getName() {
        return 'LoopMaze';
    }
}
