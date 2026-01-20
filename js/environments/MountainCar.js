/**
 * Mountain Car Environment
 * Ported from the RL-Library MountainCar implementation
 *
 * A car must build momentum by going back and forth to reach the goal
 * at the top of a hill.
 */
import { Environment } from './Environment.js';

export class MountainCar extends Environment {
    constructor(options = {}) {
        super();

        // State bounds - EXACT from Java MountainCar
        this.positionMin = options.positionMin ?? -1.2;
        this.positionMax = options.positionMax ?? 0.6;  // Java: 0.6, NOT 0.5
        this.velocityMin = options.velocityMin ?? -0.07;
        this.velocityMax = options.velocityMax ?? 0.07;

        // Goal position
        this.goalPosition = options.goalPosition ?? 0.5;

        // Start position range - EXACT from Java startRegionLftBorder and startRegionRtBorder
        this.startPositionMin = options.startPositionMin ?? -0.7;
        this.startPositionMax = options.startPositionMax ?? -0.3;

        // Default start position - EXACT from Java MountainCarState line 51
        this.defaultInitPosition = options.defaultInitPosition ?? -0.5;
        this.defaultInitVelocity = options.defaultInitVelocity ?? 0.0;

        // Random starts - EXACT from Java getDefaultParameters() line 189
        this.randomStarts = options.randomStarts ?? true;

        // Physics parameters
        this.gravity = options.gravity ?? 0.0025;
        this.carAccel = options.carAccel ?? 0.001;

        // Actions: 0=left, 1=neutral, 2=right
        this.numActions = 3;

        // Observation ranges
        this.obsRanges = [
            [this.positionMin, this.positionMax],
            [this.velocityMin, this.velocityMax]
        ];

        // State variables
        this.position = 0;
        this.velocity = 0;

        // Track last action for visualization
        this.lastAction = null;

        // Reward settings
        this.stepReward = options.stepReward ?? -1;
        this.goalReward = options.goalReward ?? 0;

        // Maximum steps per episode
        this.maxSteps = options.maxSteps ?? 400;
    }

    /**
     * Initialize the environment
     */
    init() {
        return {
            numActions: this.numActions,
            obsRanges: this.obsRanges,
            actionLabels: ['Left', 'Neutral', 'Right']
        };
    }

    /**
     * Start a new episode - EXACT from Java MountainCar.env_start() and MountainCarState
     */
    start() {
        if (this.randomStarts) {
            // Random start position in start region - EXACT from Java lines 134-137
            this.position = this.startPositionMin +
                Math.random() * (this.startPositionMax - this.startPositionMin);
        } else {
            // Default start position - EXACT from Java line 141
            this.position = this.defaultInitPosition;
        }
        this.velocity = this.defaultInitVelocity;

        this.episodeSteps = 0;
        this.episodeReward = 0;
        this.isTerminal = false;
        this.lastAction = null;
        this.currentObs = [this.position, this.velocity];
        return this.currentObs;
    }

    /**
     * Take a step in the environment
     */
    step(action) {
        if (this.isTerminal) {
            return { obs: this.currentObs, reward: 0, terminal: true };
        }

        this.lastAction = action;

        // Map action to acceleration direction
        const accelDir = action - 1; // -1, 0, or 1

        // Update velocity
        this.velocity += accelDir * this.carAccel -
                        this.gravity * Math.cos(3 * this.position);

        // Clip velocity
        this.velocity = Math.max(this.velocityMin,
                        Math.min(this.velocityMax, this.velocity));

        // Update position
        this.position += this.velocity;

        // Handle position bounds - EXACT from Java MountainCarState lines 105-113
        if (this.position > this.positionMax) {
            this.position = this.positionMax;
        }
        if (this.position < this.positionMin) {
            this.position = this.positionMin;
        }
        if (this.position === this.positionMin && this.velocity < 0) {
            this.velocity = 0;
        }

        // Update observation
        this.currentObs = [this.position, this.velocity];

        // Check if at goal
        const atGoal = this.position >= this.goalPosition;

        // Calculate reward
        let reward = this.stepReward;
        if (atGoal) {
            reward = this.goalReward;
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
     * Get the terrain height at a position for VISUALIZATION
     * Note: Java uses -sin(3*x) but with inverted min/max in normalization
     * which cancels out. We use sin(3*x) with correct normalization for
     * the same visual result: goal at top right, valley in middle.
     */
    _getHeight(x) {
        return Math.sin(3 * x);
    }

    /**
     * Normalize a value to [0, 1] range
     */
    _normalize(value, min, max) {
        return (value - min) / (max - min);
    }

    /**
     * Render the mountain car - exact port from MountainVizComponent.java and CarOnMountainVizComponent.java
     */
    render(ctx, width, height) {
        // White background (as in original)
        ctx.fillStyle = '#FFFFFF';
        ctx.fillRect(0, 0, width, height);

        // Calculate height range for normalization
        const numSamples = 100;
        let minHeight = Infinity;
        let maxHeight = -Infinity;
        const heights = [];

        for (let i = 0; i <= numSamples; i++) {
            const x = this.positionMin + (i / numSamples) * (this.positionMax - this.positionMin);
            const h = this._getHeight(x);
            heights.push(h);
            minHeight = Math.min(minHeight, h);
            maxHeight = Math.max(maxHeight, h);
        }

        // Draw mountain terrain - black lines (as in original)
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 2;
        ctx.beginPath();

        for (let i = 0; i <= numSamples; i++) {
            const px = (i / numSamples) * width;
            const normalizedHeight = this._normalize(heights[i], minHeight, maxHeight);
            const py = height - normalizedHeight * height * 0.8 - height * 0.1;

            if (i === 0) {
                ctx.moveTo(px, py);
            } else {
                ctx.lineTo(px, py);
            }
        }
        ctx.stroke();

        // Draw goal marker - green rectangle (as in original)
        const goalNormX = this._normalize(this.goalPosition, this.positionMin, this.positionMax);
        const goalHeight = this._getHeight(this.goalPosition);
        const goalNormY = this._normalize(goalHeight, minHeight, maxHeight);
        const goalPx = goalNormX * width;
        const goalPy = height - goalNormY * height * 0.8 - height * 0.1;

        ctx.fillStyle = '#00FF00'; // Green
        // EXACT from Java MountainVizComponent.java lines 117-118: rectWidth = 0.05/4, rectHeight = 0.1
        const goalRectWidth = 0.05 / 4;  // 0.0125 in normalized coords
        const goalRectHeight = 0.1;
        ctx.fillRect(goalPx - (goalRectWidth * width) / 2, goalPy - (goalRectHeight * height) / 2,
                     goalRectWidth * width, goalRectHeight * height);

        // Draw car - red rectangle - EXACT from Java CarOnMountainVizComponent.java lines 86-91
        const carNormX = this._normalize(this.position, this.positionMin, this.positionMax);
        const carTerrainHeight = this._getHeight(this.position);
        const carNormY = this._normalize(carTerrainHeight, minHeight, maxHeight);
        const carPx = carNormX * width;
        const carPy = height - carNormY * height * 0.8 - height * 0.1;

        ctx.fillStyle = '#FF0000'; // Red
        // Java: rectWidth = 0.05, rectHeight = 0.05 (in normalized coords)
        const carRectWidth = 0.05 * width;
        const carRectHeight = 0.05 * height;
        // Java: transX - rectWidth/2, transY - rectHeight/2 (CENTERED on terrain)
        ctx.fillRect(carPx - carRectWidth / 2, carPy - carRectHeight / 2, carRectWidth, carRectHeight);

        // Draw action indicator - cyan bar on car - EXACT from Java CarOnMountainVizComponent.java lines 96-109
        if (this.lastAction !== null) {
            ctx.fillStyle = '#00FFFF'; // Cyan
            // Java: rectWidth / 8.0 wide, full rectHeight tall
            const indicatorWidth = carRectWidth / 8.0;
            const indicatorHeight = carRectHeight;  // Full height

            // Position based on action: 0=left, 1=center, 2=right
            // Java offsets: 0, 7/16*rectWidth, 14/16*rectWidth
            let indicatorOffsetX;
            switch (this.lastAction) {
                case 0: indicatorOffsetX = 0; break;                     // Left
                case 1: indicatorOffsetX = carRectWidth * 7 / 16; break; // Center
                case 2: indicatorOffsetX = carRectWidth * 14 / 16; break; // Right
            }

            // Car is centered at (carPx, carPy), so indicator starts at carPx - carRectWidth/2
            ctx.fillRect(
                carPx - carRectWidth / 2 + indicatorOffsetX,
                carPy - carRectHeight / 2,
                indicatorWidth,
                indicatorHeight
            );
        }
    }

    /**
     * Get action labels
     */
    getActionLabels() {
        return ['Left', 'Neutral', 'Right'];
    }

    /**
     * Get environment name
     */
    getName() {
        return 'MountainCar';
    }
}
