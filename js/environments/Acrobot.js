/**
 * Acrobot Environment
 * EXACT port from AcrobotState.java in the RL-Library
 *
 * The acrobot is a 2-link pendulum with only the second joint actuated.
 * The goal is to swing the end of the lower link up to a height above the base.
 */
import { Environment } from './Environment.js';

export class Acrobot extends Environment {
    constructor(options = {}) {
        super();

        // STATIC CONSTANTS - EXACT from Java AcrobotState.java lines 16-31
        this.numActions = 3;
        this.maxTheta1 = Math.PI;
        this.maxTheta2 = Math.PI;
        this.maxTheta1Dot = 4 * Math.PI;
        this.maxTheta2Dot = 9 * Math.PI;

        // Physical parameters - EXACT from Java
        this.m1 = 1.0;  // mass of link 1
        this.m2 = 1.0;  // mass of link 2
        this.l1 = 1.0;  // length of link 1
        this.l2 = 1.0;  // length of link 2
        this.lc1 = 0.5; // position of center of mass of link 1
        this.lc2 = 0.5; // position of center of mass of link 2
        this.I1 = 1.0;  // moment of inertia of link 1
        this.I2 = 1.0;  // moment of inertia of link 2
        this.g = 9.8;   // gravity
        this.dt = 0.05; // time step for physics integration

        // Goal - EXACT from Java line 31
        this.acrobotGoalPosition = 1.0;

        // Configuration options
        this.randomStarts = options.randomStarts ?? false;
        this.transitionNoise = options.transitionNoise ?? 0.0;

        // Observation ranges - EXACT from Java Acrobot.java lines 75-78
        this.obsRanges = [
            [-this.maxTheta1, this.maxTheta1],
            [-this.maxTheta2, this.maxTheta2],
            [-this.maxTheta1Dot, this.maxTheta1Dot],
            [-this.maxTheta2Dot, this.maxTheta2Dot]
        ];

        // State variables
        this.theta1 = 0;
        this.theta2 = 0;
        this.theta1Dot = 0;
        this.theta2Dot = 0;

        // Track last action for visualization
        this.lastAction = null;

        // Reward settings - EXACT from Java Acrobot.java lines 116-122
        this.stepReward = -1;
        this.goalReward = 0;
    }

    /**
     * Initialize the environment
     */
    init() {
        return {
            numActions: this.numActions,
            obsRanges: this.obsRanges,
            actionLabels: ['Left', 'None', 'Right']
        };
    }

    /**
     * Start a new episode - EXACT from Java AcrobotState.reset()
     */
    start() {
        this.lastAction = 0;

        if (this.randomStarts) {
            // EXACT from Java AcrobotState.resetRandom() lines 181-186
            this.theta1 = Math.random() - 0.5;
            this.theta2 = Math.random() - 0.5;
            this.theta1Dot = Math.random() - 0.5;
            this.theta2Dot = Math.random() - 0.5;
        } else {
            // EXACT from Java AcrobotState.resetBottom() lines 188-191
            this.theta1 = 0.0;
            this.theta2 = 0.0;
            this.theta1Dot = 0.0;
            this.theta2Dot = 0.0;
        }

        this.episodeSteps = 0;
        this.episodeReward = 0;
        this.isTerminal = false;
        this.currentObs = [this.theta1, this.theta2, this.theta1Dot, this.theta2Dot];
        return this.currentObs;
    }

    /**
     * Check if terminal - EXACT from Java AcrobotState.isTerminal() lines 130-139
     */
    _isTerminal() {
        // EXACT from Java - using the "New Code" version
        const firstJointEndHeight = this.l1 * Math.cos(this.theta1);
        const secondJointEndHeight = this.l2 * Math.sin(Math.PI / 2 - this.theta1 - this.theta2);
        const feetHeight = -(firstJointEndHeight + secondJointEndHeight);
        return feetHeight > this.acrobotGoalPosition;
    }

    /**
     * Take a step in the environment - EXACT from Java AcrobotState.update()
     */
    step(action) {
        if (this.isTerminal) {
            return { obs: this.currentObs, reward: 0, terminal: true };
        }

        this.lastAction = action;

        // EXACT from Java line 76
        let torque = action - 1.0;

        // EXACT from Java lines 87-89: add noise
        const theNoise = this.transitionNoise * 2.0 * (Math.random() - 0.5);
        torque += theNoise;

        // EXACT from Java: run 4 sub-steps (lines 91-109)
        let count = 0;
        while (!this._isTerminal() && count < 4) {
            count++;

            // EXACT from Java lines 95-96
            const d1 = this.m1 * Math.pow(this.lc1, 2) +
                       this.m2 * (Math.pow(this.l1, 2) + Math.pow(this.lc2, 2) +
                                  2 * this.l1 * this.lc2 * Math.cos(this.theta2)) +
                       this.I1 + this.I2;
            const d2 = this.m2 * (Math.pow(this.lc2, 2) +
                                  this.l1 * this.lc2 * Math.cos(this.theta2)) + this.I2;

            // EXACT from Java lines 98-99
            const phi_2 = this.m2 * this.lc2 * this.g *
                          Math.cos(this.theta1 + this.theta2 - Math.PI / 2.0);
            const phi_1 = -(this.m2 * this.l1 * this.lc2 * Math.pow(this.theta2Dot, 2) *
                           Math.sin(this.theta2) -
                           2 * this.m2 * this.l1 * this.lc2 * this.theta1Dot *
                           this.theta2Dot * Math.sin(this.theta2)) +
                          (this.m1 * this.lc1 + this.m2 * this.l1) * this.g *
                          Math.cos(this.theta1 - Math.PI / 2.0) + phi_2;

            // EXACT from Java lines 101-102
            const theta2_ddot = (torque + (d2 / d1) * phi_1 -
                                this.m2 * this.l1 * this.lc2 *
                                Math.pow(this.theta1Dot, 2) * Math.sin(this.theta2) - phi_2) /
                               (this.m2 * Math.pow(this.lc2, 2) + this.I2 -
                                Math.pow(d2, 2) / d1);
            const theta1_ddot = -(d2 * theta2_ddot + phi_1) / d1;

            // EXACT from Java lines 104-108: Euler integration
            this.theta1Dot += theta1_ddot * this.dt;
            this.theta2Dot += theta2_ddot * this.dt;
            this.theta1 += this.theta1Dot * this.dt;
            this.theta2 += this.theta2Dot * this.dt;
        }

        // EXACT from Java lines 110-116: clip angular velocities
        if (Math.abs(this.theta1Dot) > this.maxTheta1Dot) {
            this.theta1Dot = Math.sign(this.theta1Dot) * this.maxTheta1Dot;
        }
        if (Math.abs(this.theta2Dot) > this.maxTheta2Dot) {
            this.theta2Dot = Math.sign(this.theta2Dot) * this.maxTheta2Dot;
        }

        // EXACT from Java lines 120-127: hard constraint on thetas
        if (Math.abs(this.theta2) > Math.PI) {
            this.theta2 = Math.sign(this.theta2) * Math.PI;
            this.theta2Dot = 0;
        }
        if (Math.abs(this.theta1) > Math.PI) {
            this.theta1 = Math.sign(this.theta1) * Math.PI;
            this.theta1Dot = 0;
        }

        // Update observation
        this.currentObs = [this.theta1, this.theta2, this.theta1Dot, this.theta2Dot];

        // Check terminal and calculate reward - EXACT from Java Acrobot.java lines 115-123
        let reward = this.stepReward;
        if (this._isTerminal()) {
            reward = this.goalReward;
            this.isTerminal = true;
        }

        // Update statistics
        this.episodeSteps++;
        this.totalSteps++;
        this.episodeReward += reward;

        return {
            obs: this.currentObs,
            reward: reward,
            terminal: this.isTerminal
        };
    }

    /**
     * Render the acrobot - EXACT from Java AcrobotBotComponent.java
     */
    render(ctx, width, height) {
        // White background - EXACT from Java line 34-35
        ctx.fillStyle = '#FFFFFF';
        ctx.fillRect(0, 0, width, height);

        // Scale to fit canvas (Java uses 100x100 coordinate system)
        const scale = Math.min(width, height) / 100;
        ctx.save();
        ctx.scale(scale, scale);

        // Constants from Java lines 19-25
        const joint1X = 50;
        const joint1Y = 30;
        const leg1length = 25;
        const leg2length = leg1length;
        const circleSize1 = 6;
        const circleSize2 = 4;
        const circleSize3 = 2;

        // Draw goal line - green - EXACT from Java lines 38-40
        const goalY = joint1Y - leg1length;
        ctx.strokeStyle = '#00FF00';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, goalY);
        ctx.lineTo(100, goalY);
        ctx.stroke();

        // Calculate joint positions - EXACT from Java lines 43-44
        const joint2X = leg1length * Math.sin(this.theta1) + joint1X;
        const joint2Y = leg1length * Math.cos(this.theta1) + joint1Y;

        // Draw first link - black - EXACT from Java line 45
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(joint1X, joint1Y);
        ctx.lineTo(joint2X, joint2Y);
        ctx.stroke();

        // Draw first joint circle - blue - EXACT from Java lines 47-48
        ctx.fillStyle = '#0000FF';
        ctx.beginPath();
        ctx.arc(joint1X, joint1Y, circleSize1 / 2, 0, 2 * Math.PI);
        ctx.fill();

        // Calculate foot position - EXACT from Java lines 50-51
        const joint3X = leg2length * Math.cos(Math.PI / 2 - this.theta2 - this.theta1) + joint2X;
        const joint3Y = leg2length * Math.sin(Math.PI / 2 - this.theta1 - this.theta2) + joint2Y;

        // Draw second link - black - EXACT from Java lines 52-53
        ctx.strokeStyle = '#000000';
        ctx.beginPath();
        ctx.moveTo(joint2X, joint2Y);
        ctx.lineTo(joint3X, joint3Y);
        ctx.stroke();

        // Draw second joint circle - blue - EXACT from Java lines 55-56
        ctx.fillStyle = '#0000FF';
        ctx.beginPath();
        ctx.arc(joint2X, joint2Y, circleSize2 / 2, 0, 2 * Math.PI);
        ctx.fill();

        // Draw feet - cyan - EXACT from Java lines 61-62
        ctx.fillStyle = '#00FFFF';
        ctx.beginPath();
        ctx.arc(joint3X, joint3Y, circleSize3 / 2, 0, 2 * Math.PI);
        ctx.fill();

        ctx.restore();
    }

    /**
     * Get action labels
     */
    getActionLabels() {
        return ['Left', 'None', 'Right'];
    }

    /**
     * Get environment name
     */
    getName() {
        return 'Acrobot';
    }
}
