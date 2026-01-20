/**
 * Base Environment Class
 * Abstract base class for reinforcement learning environments.
 */
export class Environment {
    constructor() {
        this.numActions = 0;
        this.obsRanges = [];
        this.currentObs = null;
        this.episodeSteps = 0;
        this.totalSteps = 0;
        this.episodeReward = 0;
        this.isTerminal = false;
    }

    /**
     * Initialize the environment
     * @returns {Object} Task specification
     */
    init() {
        throw new Error('Subclasses must implement init()');
    }

    /**
     * Start a new episode
     * @returns {number[]} Initial observation
     */
    start() {
        throw new Error('Subclasses must implement start()');
    }

    /**
     * Take a step in the environment
     * @param {number} action - Action to take
     * @returns {{obs: number[], reward: number, terminal: boolean}}
     */
    step(action) {
        throw new Error('Subclasses must implement step()');
    }

    /**
     * Get the current observation
     * @returns {number[]}
     */
    getObs() {
        return this.currentObs;
    }

    /**
     * Get the number of actions
     * @returns {number}
     */
    getNumActions() {
        return this.numActions;
    }

    /**
     * Get observation ranges
     * @returns {number[][]} Array of [min, max] for each dimension
     */
    getObsRanges() {
        return this.obsRanges;
    }

    /**
     * Check if episode is terminal
     * @returns {boolean}
     */
    isEpisodeTerminal() {
        return this.isTerminal;
    }

    /**
     * Get episode statistics
     */
    getEpisodeStats() {
        return {
            steps: this.episodeSteps,
            reward: this.episodeReward
        };
    }

    /**
     * Render the environment (to be overridden)
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {number} width - Canvas width
     * @param {number} height - Canvas height
     */
    render(ctx, width, height) {
        // Default: clear canvas
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, width, height);
    }

    /**
     * Get action labels for display
     * @returns {string[]}
     */
    getActionLabels() {
        return Array.from({ length: this.numActions }, (_, i) => `Action ${i}`);
    }

    /**
     * Get environment name
     * @returns {string}
     */
    getName() {
        return 'Environment';
    }
}
