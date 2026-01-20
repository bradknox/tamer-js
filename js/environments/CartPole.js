/**
 * CartPole (Inverted Pendulum) Environment
 * EXACT port from CartPoleTransModel.java in the TAMER project
 *
 * Balance a pole on a cart by moving the cart left or right.
 */
import { Environment } from './Environment.js';

export class CartPole extends Environment {
    constructor(options = {}) {
        super();

        // Physical parameters - EXACT from Java
        this.GRAVITY = 9.8;
        this.MASSCART = 1.0;
        this.MASSPOLE = 0.1;
        this.TOTAL_MASS = this.MASSPOLE + this.MASSCART;
        this.LENGTH = 0.5;  // Actually half the pole's length - EXACT from Java
        this.POLEMASS_LENGTH = this.MASSPOLE * this.LENGTH;
        this.FORCE_MAG = 10.0;
        this.TAU = 0.02;  // Seconds between state updates - EXACT from Java
        this.FOURTHIRDS = 4.0 / 3.0;

        // Bounds - EXACT from Java
        this.leftCartBound = -2.4;
        this.rightCartBound = 2.4;
        this.leftAngleBound = -Math.PI * 12.0 / 180.0;  // -12 degrees in radians
        this.rightAngleBound = Math.PI * 12.0 / 180.0;  // 12 degrees in radians

        // Random start and noise settings - EXACT from Java defaults
        this.randomStartStates = options.randomStartStates ?? true;
        this.transitionNoise = options.transitionNoise ?? 1.0;  // EXACT from Java line 36

        // Actions: 0=left, 1=right
        this.numActions = 2;

        // Observation ranges: [x, x_dot, theta, theta_dot]
        // Using wider ranges for function approximation
        this.obsRanges = [
            [this.leftCartBound, this.rightCartBound],
            [-3.0, 3.0],  // velocity
            [this.leftAngleBound, this.rightAngleBound],
            [-3.0, 3.0]   // angular velocity
        ];

        // State variables
        this.x = 0;        // cart position
        this.x_dot = 0;    // cart velocity
        this.theta = 0;    // pole angle
        this.theta_dot = 0; // pole angular velocity

        // Track last action for visualization
        this.lastAction = null;

        // Reward settings - EXACT from Java getReward()
        this.stepReward = options.stepReward ?? 1.0;  // +1 when not failing
        this.failReward = options.failReward ?? -1.0; // -1 when failing

        // Maximum steps per episode
        this.maxSteps = options.maxSteps ?? 500;
    }

    /**
     * Initialize the environment
     */
    init() {
        return {
            numActions: this.numActions,
            obsRanges: this.obsRanges,
            actionLabels: ['Left', 'Right']
        };
    }

    /**
     * Start a new episode - EXACT from Java getStartObs()
     */
    start() {
        if (this.randomStartStates) {
            // Random start states near equilibrium - EXACT from Java
            this.x = Math.random() - 0.5;                    // x: [-0.5, 0.5]
            this.x_dot = Math.random() - 0.5;                // x_dot: [-0.5, 0.5]
            this.theta = (Math.random() - 0.5) / 8.0;        // theta: [-0.0625, 0.0625]
            this.theta_dot = (Math.random() - 0.5) / 8.0;    // theta_dot: [-0.0625, 0.0625]
        } else {
            // Default start at equilibrium
            this.x = 0;
            this.x_dot = 0;
            this.theta = 0;
            this.theta_dot = 0;
        }

        this.episodeSteps = 0;
        this.episodeReward = 0;
        this.isTerminal = false;
        this.lastAction = null;
        this.currentObs = [this.x, this.x_dot, this.theta, this.theta_dot];
        return this.currentObs;
    }

    /**
     * Take a step in the environment - EXACT from Java sampleNextObsNoForceCont()
     */
    step(action) {
        if (this.isTerminal) {
            return { obs: this.currentObs, reward: 0, terminal: true };
        }

        this.lastAction = action;

        // Check for terminal state BEFORE applying action - EXACT from Java inFailure()
        // Java uses <= and >= (at boundary = failure)
        if (this.x <= this.leftCartBound || this.x >= this.rightCartBound ||
            this.theta <= this.leftAngleBound || this.theta >= this.rightAngleBound) {
            this.isTerminal = true;
            return {
                obs: this.currentObs,
                reward: this.failReward,
                terminal: true
            };
        }

        // Apply force based on action - EXACT from Java
        let force;
        if (action > 0) {
            force = this.FORCE_MAG;
        } else {
            force = -this.FORCE_MAG;
        }

        // Add transition noise - EXACT from Java lines 110-112
        // Noise of 1.0 means possibly full opposite action
        const thisNoise = 2.0 * this.transitionNoise * this.FORCE_MAG * (Math.random() - 0.5);
        force += thisNoise;

        // Physics calculations - EXACT from Java
        const costheta = Math.cos(this.theta);
        const sintheta = Math.sin(this.theta);

        const temp = (force + this.POLEMASS_LENGTH * this.theta_dot * this.theta_dot * sintheta) / this.TOTAL_MASS;

        const thetaacc = (this.GRAVITY * sintheta - costheta * temp) /
                        (this.LENGTH * (this.FOURTHIRDS - this.MASSPOLE * costheta * costheta / this.TOTAL_MASS));

        const xacc = temp - this.POLEMASS_LENGTH * thetaacc * costheta / this.TOTAL_MASS;

        // Update state using Euler's method - EXACT from Java
        this.x += this.TAU * this.x_dot;
        this.x_dot += this.TAU * xacc;
        this.theta += this.TAU * this.theta_dot;
        this.theta_dot += this.TAU * thetaacc;

        // Normalize theta to [-π, π] - EXACT from Java
        while (this.theta >= Math.PI) {
            this.theta -= 2.0 * Math.PI;
        }
        while (this.theta < -Math.PI) {
            this.theta += 2.0 * Math.PI;
        }

        // Clamp at boundaries - EXACT from Java
        if (this.x < this.leftCartBound) {
            this.x = this.leftCartBound;
        } else if (this.x > this.rightCartBound) {
            this.x = this.rightCartBound;
        }
        if (this.theta < this.leftAngleBound) {
            this.theta = this.leftAngleBound;
        } else if (this.theta > this.rightAngleBound) {
            this.theta = this.rightAngleBound;
        }

        // Update observation
        this.currentObs = [this.x, this.x_dot, this.theta, this.theta_dot];

        // Calculate reward
        const reward = this.stepReward;

        // Update statistics
        this.episodeSteps++;
        this.totalSteps++;
        this.episodeReward += reward;

        // Check max steps (success!)
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
     * Check if in failure state - EXACT from Java inFailure()
     * Java uses <= and >= (at boundary = failure)
     */
    inFailure() {
        return (this.x <= this.leftCartBound || this.x >= this.rightCartBound ||
                this.theta <= this.leftAngleBound || this.theta >= this.rightAngleBound);
    }

    /**
     * Normalize a value to [0, 1] range
     */
    _normalize(value, min, max) {
        return (value - min) / (max - min);
    }

    /**
     * Render the cart-pole - EXACT port from CartPoleCartComponent.java and CartPoleTrackComponent.java
     */
    render(ctx, width, height) {
        // White background - EXACT from Java CartPoleTrackComponent line 24-25
        ctx.fillStyle = '#FFFFFF';
        ctx.fillRect(0, 0, width, height);

        // Coordinate system setup - EXACT from Java: track at y=0.88, x from 0.10 to 0.90
        const trackY = height * 0.88;
        const trackLeftX = width * 0.10;
        const trackRightX = width * 0.90;

        // Draw track (black line) - EXACT from Java CartPoleTrackComponent line 32
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(trackLeftX, trackY);
        ctx.lineTo(trackRightX, trackY);
        ctx.stroke();

        // Calculate scale and cart position - EXACT from Java CartPoleCartComponent lines 48-49
        // Java: transX = normalize(x, leftBound, rightBound) * 0.8 + 0.1
        const normalizedX = this._normalize(this.x, this.leftCartBound, this.rightCartBound);
        const cartX = normalizedX * (trackRightX - trackLeftX) + trackLeftX;

        // Cart dimensions - EXACT from Java CartPoleCartComponent lines 46, 52
        // Java uses twentyPercent (0.2) for width, fivePercent (0.05) for height
        const cartWidth = width * 0.20;
        const cartHeight = width * 0.05;

        // Cart Y position - EXACT from Java: transY = eightyPercent (0.8), not at track
        // Cart TOP is at 80%, cart BOTTOM is at 80% + 5% = 85% (above track at 88%)
        const cartY = height * 0.80;

        // Draw cart - blue rectangle - EXACT from Java line 51-53
        ctx.fillStyle = '#0000FF';
        ctx.fillRect(cartX - cartWidth / 2, cartY, cartWidth, cartHeight);

        // Draw wheels - red ovals - EXACT from Java drawWheels() lines 98-106
        // Java: wheelRad = carRect.height, wheels at carMidY (center Y of cart)
        // Java fillOval places TOP-LEFT, so wheel top is at carMidY
        ctx.fillStyle = '#FF0000';
        const wheelRad = cartHeight;
        const carX1 = cartX - cartWidth / 4;  // 1/4 from left edge
        const carX2 = cartX + cartWidth / 4;  // 1/4 from right edge
        const carMidY = cartY + cartHeight / 2;  // Center Y of cart

        // Java uses fillOval(x - wheelRad/2, carMidY, wheelRad, wheelRad)
        // So wheel TOP is at carMidY, wheel extends DOWN from there
        ctx.beginPath();
        ctx.ellipse(carX1, carMidY + wheelRad / 2, wheelRad / 2, wheelRad / 2, 0, 0, Math.PI * 2);
        ctx.fill();
        ctx.beginPath();
        ctx.ellipse(carX2, carMidY + wheelRad / 2, wheelRad / 2, wheelRad / 2, 0, 0, Math.PI * 2);
        ctx.fill();

        // Pole parameters - EXACT from Java: poleLength = 0.3 (30% of screen)
        // Pole starts at cart position (transX, transY) which is cart TOP
        const poleLength = height * 0.3;
        const poleStartX = cartX;
        const poleStartY = cartY;  // Pole starts at cart TOP

        // Translate angle for rendering - EXACT from Java translateAngle(): angle - PI/2
        // This makes the pole point UP when theta=0
        const renderAngle = this.theta - Math.PI / 2;

        // Draw failure angle lines - RED - EXACT from Java lines 67-74
        ctx.strokeStyle = '#FF0000';
        ctx.lineWidth = 2;
        const minRenderAngle = this.leftAngleBound - Math.PI / 2;
        const maxRenderAngle = this.rightAngleBound - Math.PI / 2;

        // Left failure line
        const failLeftX = poleStartX + poleLength * Math.cos(minRenderAngle);
        const failLeftY = poleStartY + poleLength * Math.sin(minRenderAngle);
        ctx.beginPath();
        ctx.moveTo(poleStartX, poleStartY);
        ctx.lineTo(failLeftX, failLeftY);
        ctx.stroke();

        // Right failure line
        const failRightX = poleStartX + poleLength * Math.cos(maxRenderAngle);
        const failRightY = poleStartY + poleLength * Math.sin(maxRenderAngle);
        ctx.beginPath();
        ctx.moveTo(poleStartX, poleStartY);
        ctx.lineTo(failRightX, failRightY);
        ctx.stroke();

        // Draw pole - black line - EXACT from Java lines 57-63
        const poleEndX = poleStartX + poleLength * Math.cos(renderAngle);
        const poleEndY = poleStartY + poleLength * Math.sin(renderAngle);

        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 6;
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(poleStartX, poleStartY);
        ctx.lineTo(poleEndX, poleEndY);
        ctx.stroke();

        // Draw pivot point
        ctx.fillStyle = '#000000';
        ctx.beginPath();
        ctx.arc(poleStartX, poleStartY, 4, 0, Math.PI * 2);
        ctx.fill();

        // Draw action indicator - cyan rectangle on cart - EXACT from Java lines 77-91
        // Java: width = carRect.width / 10.0, height = carRect.height (full)
        // Java offsets: action 0 -> 0, action 1 -> 0.9 * carRect.width
        if (this.lastAction !== null) {
            ctx.fillStyle = '#00FFFF';
            const indicatorWidth = cartWidth / 10.0;
            const indicatorHeight = cartHeight;  // Full height

            let indicatorOffsetX;
            if (this.lastAction === 0) {
                indicatorOffsetX = 0;
            } else {
                indicatorOffsetX = 0.9 * cartWidth;
            }

            ctx.fillRect(
                cartX - cartWidth / 2 + indicatorOffsetX,
                cartY,  // Indicator on cart, which starts at cartY
                indicatorWidth,
                indicatorHeight
            );
        }
    }

    /**
     * Get action labels
     */
    getActionLabels() {
        return ['Left', 'Right'];
    }

    /**
     * Get environment name
     */
    getName() {
        return 'CartPole';
    }
}
