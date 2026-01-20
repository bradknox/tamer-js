/**
 * RobotArm - Multi-jointed robot arm environment
 * EXACT port from RobotArm.java and RobotArmState.java in the TAMER project
 *
 * A robot arm with multiple joints that must reach a target location.
 * The agent controls joint angles to move the end effector to the goal.
 *
 * @author Brad Knox
 */

export class RobotArm {
    // Static constants - EXACT from Java
    static worldDims = [20, 20, 10, 10];  // Discretization of joint angles
    static origin = { x: 0.5, y: 1.0 };
    static SEG_LENS = [0.3, 0.1, 0.2, 0.1, 0.2, 0.05, 0.05];  // Length of each arm segment
    static graspRadius = 0.05;
    static targetLoc = { x: 0.8, y: 0.6 };  // Goal location
    static rewardPerStep = -1.0;
    static rewardAtGoal = 0.0;
    static agentSpeed = 1;

    constructor(options = {}) {
        this.randomStarts = options.randomStarts ?? false;
        this.transitionNoise = options.transitionNoise ?? 0.0;

        // Agent state
        this.agentPosition = null;
        this.agentPrevPosition = null;

        // Default initial position
        this.defaultInitPosition = options.defaultInitPosition ?? null;
        this.overridingDefInitPosition = options.overridingDefInitPosition ?? null;

        this.lastAction = 0;

        // Initialize position
        this.reset();
    }

    /**
     * Get environment name
     */
    getName() {
        return 'RobotArm';
    }

    /**
     * Initialize environment and return spec - EXACT from Java env_init()
     */
    init() {
        const numActions = RobotArm.worldDims.length * 2;

        return {
            obsRanges: RobotArm.worldDims.map(dim => [0, dim - 1]),
            numActions: numActions,
            discountFactor: 1.0,
            isEpisodic: true,
            rewardRange: [-80, 0]
        };
    }

    /**
     * Start a new episode - EXACT from Java env_start()
     */
    start() {
        this.reset();
        return this._makeObservation();
    }

    /**
     * Take a step - EXACT from Java env_step()
     * @param {number} action - Action to take (0 to numActions-1)
     * @returns {{obs: number[], reward: number, terminal: boolean}}
     */
    step(action) {
        const maxPossibleAct = (RobotArm.worldDims.length * 2) - 1;
        if (action > maxPossibleAct || action < 0) {
            console.error("Invalid action selected in RobotArm: " + action);
            action = Math.floor(Math.random() * (maxPossibleAct + 1));
        }

        this._update(action);

        return {
            obs: this._makeObservation(),
            reward: this._getReward(),
            terminal: this._inGoalRegion()
        };
    }

    /**
     * Reset state - EXACT from Java reset()
     */
    reset() {
        this.agentPosition = this._sampleEnvStart(true);
        if (this.agentPrevPosition !== null) {
            this.agentPrevPosition = this.agentPrevPosition.slice();
        } else {
            this.agentPrevPosition = this.agentPosition.slice();
        }
    }

    /**
     * Sample environment start state - EXACT from Java sampleEnvStart()
     */
    _sampleEnvStart(envAsking) {
        let agentLoc;

        if ((this.overridingDefInitPosition === null || !envAsking) &&
            this.defaultInitPosition !== null) {
            agentLoc = this.defaultInitPosition.slice();
        } else if (this.overridingDefInitPosition !== null) {
            agentLoc = this.overridingDefInitPosition.slice();
        } else {
            agentLoc = this._getRandomStartState();
        }

        if (this.randomStarts) {
            agentLoc = this._getRandomStartState();
        }

        return agentLoc;
    }

    /**
     * Get random start state - EXACT from Java getRandomStartState()
     */
    _getRandomStartState() {
        let agentLoc;
        do {
            agentLoc = new Array(RobotArm.worldDims.length);
            for (let i = 0; i < agentLoc.length; i++) {
                agentLoc[i] = Math.floor(Math.random() * RobotArm.worldDims[i]);
            }
        } while (RobotArm.inGoalRegion(agentLoc) || !RobotArm.isStateLegal(agentLoc));
        return agentLoc;
    }

    /**
     * Update state with action - EXACT from Java update()
     */
    _update(action) {
        this.lastAction = action;
        this.agentPrevPosition = this.agentPosition.slice();

        const nextAgentLoc = this.agentPosition.slice();

        const locChange = (action % 2 === 0) ? RobotArm.agentSpeed : (-1 * RobotArm.agentSpeed);
        nextAgentLoc[Math.floor(action / 2)] += locChange;

        if (RobotArm.isMoveLegal(this.agentPosition, nextAgentLoc)) {
            this.agentPosition = nextAgentLoc;
        }
        // else no change
    }

    /**
     * Get joint 2D locations - EXACT from Java getJointLocs()
     * @param {number[]} relativeJointAngs - Joint angles
     * @returns {Array<{x: number, y: number}>} Joint locations
     */
    static getJointLocs(relativeJointAngs) {
        const jointLocs = [{ x: RobotArm.origin.x, y: RobotArm.origin.y }];

        let lastJointLoc = { x: RobotArm.origin.x, y: RobotArm.origin.y };
        let lastJointAng = 0;

        for (let segI = 0; segI < RobotArm.worldDims.length; segI++) {
            let jointAng;
            if (segI === 0) {
                jointAng = (((0.5 + relativeJointAngs[segI]) / RobotArm.worldDims[segI]) * Math.PI) + Math.PI;
            } else {
                jointAng = lastJointAng + ((relativeJointAngs[segI] / (RobotArm.worldDims[segI] - 1.0)) * 2 * Math.PI);
            }

            const newJointLoc = {
                x: lastJointLoc.x + (Math.cos(jointAng) * RobotArm.SEG_LENS[segI]),
                y: lastJointLoc.y + (Math.sin(jointAng) * RobotArm.SEG_LENS[segI])
            };

            jointLocs.push(newJointLoc);

            lastJointLoc = newJointLoc;
            lastJointAng = jointAng;
        }

        return jointLocs;
    }

    /**
     * Get reward - EXACT from Java getReward()
     */
    _getReward() {
        return RobotArm.getReward(this.agentPosition);
    }

    /**
     * Static reward calculation - EXACT from Java
     */
    static getReward(relativeJointAngs) {
        if (RobotArm.inGoalRegion(relativeJointAngs)) {
            return RobotArm.rewardAtGoal;
        } else {
            return RobotArm.rewardPerStep;
        }
    }

    /**
     * Check if in goal region - EXACT from Java inGoalRegion()
     */
    _inGoalRegion() {
        return RobotArm.inGoalRegion(this.agentPosition);
    }

    /**
     * Static goal check - EXACT from Java
     */
    static inGoalRegion(relativeJointAngs) {
        const jointLocs = RobotArm.getJointLocs(relativeJointAngs);
        const endEffectorLoc = jointLocs[jointLocs.length - 1];

        const distSqrdToGoal = Math.pow(Math.abs(endEffectorLoc.x - RobotArm.targetLoc.x), 2) +
                               Math.pow(Math.abs(endEffectorLoc.y - RobotArm.targetLoc.y), 2);
        const distToTarget = Math.sqrt(distSqrdToGoal);

        return distToTarget < RobotArm.graspRadius;
    }

    /**
     * Check if state is legal - EXACT from Java isStateLegal()
     */
    static isStateLegal(jointAngs) {
        // Are joint angles in specified range?
        for (let i = 0; i < jointAngs.length; i++) {
            if (jointAngs[i] < 0 || jointAngs[i] >= RobotArm.worldDims[i]) {
                return false;
            }
        }

        // Are all 2D joint locations in bounds?
        const jointLocs = RobotArm.getJointLocs(jointAngs);
        for (const jointLoc of jointLocs) {
            if (jointLoc.x > 1 || jointLoc.x < 0 ||
                jointLoc.y > 1 || jointLoc.y < 0) {
                return false;
            }
        }

        return true;
    }

    /**
     * Check if move is legal - EXACT from Java isMoveLegal()
     */
    static isMoveLegal(currJointAngs, nextJointAngs) {
        return RobotArm.isStateLegal(nextJointAngs);
    }

    /**
     * Make observation - EXACT from Java makeObservation()
     */
    _makeObservation() {
        return this.agentPosition.slice();
    }

    /**
     * Get current state for visualization
     */
    getState() {
        return {
            position: this.agentPosition.slice(),
            prevPosition: this.agentPrevPosition ? this.agentPrevPosition.slice() : null,
            jointLocs: RobotArm.getJointLocs(this.agentPosition),
            targetLoc: RobotArm.targetLoc,
            graspRadius: RobotArm.graspRadius,
            inGoal: this._inGoalRegion()
        };
    }

    /**
     * Get position
     */
    getPosition() {
        return this.agentPosition.slice();
    }

    /**
     * Get previous position
     */
    getPrevPosition() {
        return this.agentPrevPosition ? this.agentPrevPosition.slice() : null;
    }

    /**
     * Get last action
     */
    getLastAction() {
        return this.lastAction;
    }

    /**
     * Get action labels for display
     */
    getActionLabels() {
        const labels = [];
        for (let i = 0; i < RobotArm.worldDims.length; i++) {
            labels.push(`J${i}+`);
            labels.push(`J${i}-`);
        }
        return labels;
    }

    /**
     * Get episode stats (for compatibility with TamerApp)
     */
    getEpisodeStats() {
        return {
            steps: 0,
            inGoal: this._inGoalRegion()
        };
    }

    /**
     * Render the environment
     */
    render(ctx, canvasWidth, canvasHeight) {
        const state = this.getState();
        const { jointLocs, targetLoc, graspRadius, inGoal } = state;

        // Scale factor to fit the [0,1] x [0,1] world into canvas
        const scale = Math.min(canvasWidth, canvasHeight) * 0.9;
        const offsetX = (canvasWidth - scale) / 2;
        const offsetY = (canvasHeight - scale) / 2;

        const toCanvasX = (x) => offsetX + x * scale;
        const toCanvasY = (y) => offsetY + (1 - y) * scale;  // Flip Y axis

        // Clear background
        ctx.fillStyle = '#f0f0f0';
        ctx.fillRect(0, 0, canvasWidth, canvasHeight);

        // Draw boundary
        ctx.strokeStyle = '#666666';
        ctx.lineWidth = 2;
        ctx.strokeRect(offsetX, offsetY, scale, scale);

        // Draw target
        ctx.beginPath();
        ctx.arc(toCanvasX(targetLoc.x), toCanvasY(targetLoc.y), graspRadius * scale, 0, Math.PI * 2);
        ctx.fillStyle = inGoal ? '#00ff00' : '#ff6666';
        ctx.fill();
        ctx.strokeStyle = '#cc0000';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Draw arm segments
        ctx.strokeStyle = '#333333';
        ctx.lineWidth = 4;
        ctx.lineCap = 'round';

        for (let i = 0; i < jointLocs.length - 1; i++) {
            const start = jointLocs[i];
            const end = jointLocs[i + 1];

            ctx.beginPath();
            ctx.moveTo(toCanvasX(start.x), toCanvasY(start.y));
            ctx.lineTo(toCanvasX(end.x), toCanvasY(end.y));
            ctx.stroke();
        }

        // Draw joints
        for (let i = 0; i < jointLocs.length; i++) {
            const joint = jointLocs[i];
            ctx.beginPath();
            ctx.arc(toCanvasX(joint.x), toCanvasY(joint.y), i === 0 ? 8 : 5, 0, Math.PI * 2);
            ctx.fillStyle = i === 0 ? '#666666' : '#3333ff';
            ctx.fill();
        }

        // Draw end effector
        const endEffector = jointLocs[jointLocs.length - 1];
        ctx.beginPath();
        ctx.arc(toCanvasX(endEffector.x), toCanvasY(endEffector.y), 8, 0, Math.PI * 2);
        ctx.fillStyle = inGoal ? '#00cc00' : '#ff9900';
        ctx.fill();
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Draw success message
        if (inGoal) {
            ctx.fillStyle = '#00aa00';
            ctx.font = 'bold 20px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('Goal Reached!', canvasWidth / 2, 30);
        }
    }
}
