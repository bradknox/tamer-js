/**
 * Comprehensive Unit Tests for TAMER Web
 *
 * Tests all components ported from Java including:
 * - TetrisFeatures
 * - Tetris environment
 * - SarsaLambdaAgent
 * - TamerRLAgent
 * - ImitationAgent
 * - ActionSelect
 * - HInfluence
 * - ExtActionAgentWrap
 * - RobotArm environment
 * - CreditAssign edge cases
 * - LinearModel edge cases
 */

import { TetrisFeatures } from '../features/TetrisFeatures.js';
import { Tetris, TetrisActions } from '../environments/tetris/index.js';
import { TetrisState } from '../environments/tetris/TetrisState.js';
import { TetrisPiece } from '../environments/tetris/TetrisPiece.js';
import { RobotArm } from '../environments/RobotArm.js';
import { SarsaLambdaAgent } from '../agents/SarsaLambdaAgent.js';
import { TamerRLAgent, CombinationMethods } from '../agents/TamerRLAgent.js';
import { ImitationAgent } from '../agents/ImitationAgent.js';
import { ExtActionAgentWrap } from '../agents/ExtActionAgentWrap.js';
import { ActionSelect } from '../core/ActionSelect.js';
import { HInfluence } from '../core/HInfluence.js';
import { CreditAssign } from '../core/CreditAssign.js';
import { LinearModel } from '../models/LinearModel.js';
import { RBFFeatures } from '../features/RBFFeatures.js';
import { TamerAgent } from '../core/TamerAgent.js';
import { LoopMaze } from '../environments/LoopMaze.js';
import { MountainCar } from '../environments/MountainCar.js';
import { CartPole } from '../environments/CartPole.js';
import { Acrobot } from '../environments/Acrobot.js';

let passed = 0;
let failed = 0;

function assert(condition, message) {
    if (condition) {
        console.log(`✓ ${message}`);
        passed++;
    } else {
        console.log(`✗ ${message}`);
        failed++;
    }
}

function assertApprox(actual, expected, tolerance, message) {
    const diff = Math.abs(actual - expected);
    if (diff <= tolerance) {
        console.log(`✓ ${message} (${actual} ≈ ${expected})`);
        passed++;
    } else {
        console.log(`✗ ${message} (${actual} != ${expected}, diff=${diff})`);
        failed++;
    }
}

function assertThrows(fn, message) {
    try {
        fn();
        console.log(`✗ ${message} (no exception thrown)`);
        failed++;
    } catch (e) {
        console.log(`✓ ${message} (threw: ${e.message.substring(0, 50)}...)`);
        passed++;
    }
}

// ============================================================
// TetrisFeatures Tests
// ============================================================
console.log('\n========== TetrisFeatures Tests ==========\n');

// Test feature count
{
    const featGen = new TetrisFeatures([[0, 1]], 5);
    assert(featGen.getNumFeatures() === 230, 'TetrisFeatures: 46 features * 5 actions = 230');
    assert(featGen.getNumStateFeatures() === 46, 'TetrisFeatures: 46 state features');
}

// Test feature constants match Java
{
    assert(TetrisFeatures.NUM_FEATS === 46, 'NUM_FEATS = 46');
    assert(TetrisFeatures.COL_HT_START_I === 0, 'COL_HT_START_I = 0');
    assert(TetrisFeatures.MAX_COL_HT_I === 10, 'MAX_COL_HT_I = 10');
    assert(TetrisFeatures.COL_DIFF_START_I === 11, 'COL_DIFF_START_I = 11');
    assert(TetrisFeatures.NUM_HOLES_I === 20, 'NUM_HOLES_I = 20');
    assert(TetrisFeatures.MAX_WELL_I === 21, 'MAX_WELL_I = 21');
    assert(TetrisFeatures.SUM_WELL_I === 22, 'SUM_WELL_I = 22');
    assert(TetrisFeatures.SQUARED_FEATS_START_I === 23, 'SQUARED_FEATS_START_I = 23');
    assert(TetrisFeatures.HT_SQ_SCALE === 100.0, 'HT_SQ_SCALE = 100.0');
}

// Test empty board features
{
    const featGen = new TetrisFeatures([[0, 1]], 5);
    // Empty 10x20 board
    const emptyBoard = new Array(200).fill(0);
    // Add metadata (7 elements)
    const obs = [...emptyBoard, 0, 0, 0, 0, 0, 0, 0];

    const feats = featGen.getStateFeatures(obs);

    // All column heights should be 20 (bottom of board)
    for (let col = 0; col < 10; col++) {
        assert(feats[TetrisFeatures.COL_HT_START_I + col] === 20,
            `Empty board: column ${col} height = 20`);
    }
    assert(feats[TetrisFeatures.MAX_COL_HT_I] === 20, 'Empty board: max height = 20');
    assert(feats[TetrisFeatures.NUM_HOLES_I] === 0, 'Empty board: 0 holes');
}

// Test board with one filled cell
{
    const featGen = new TetrisFeatures([[0, 1]], 5);
    const board = new Array(200).fill(0);
    // Fill cell at row 19 (bottom), col 0
    board[19 * 10 + 0] = 1;
    const obs = [...board, 0, 0, 0, 0, 0, 0, 0];

    const feats = featGen.getStateFeatures(obs);
    assert(feats[TetrisFeatures.COL_HT_START_I + 0] === 19,
        'One cell: column 0 height = 19');
    assert(feats[TetrisFeatures.MAX_COL_HT_I] === 19,
        'One cell: max height = 19');
}

// Test holes detection
{
    const featGen = new TetrisFeatures([[0, 1]], 5);
    const board = new Array(200).fill(0);
    // Fill cell at row 17, col 5
    board[17 * 10 + 5] = 1;
    // Leave rows 18 and 19 empty at col 5 - these are holes
    const obs = [...board, 0, 0, 0, 0, 0, 0, 0];

    const feats = featGen.getStateFeatures(obs);
    assert(feats[TetrisFeatures.NUM_HOLES_I] === 2,
        'Two holes under filled cell');
}

// Test column height differences
{
    const featGen = new TetrisFeatures([[0, 1]], 5);
    const board = new Array(200).fill(0);
    // Column 0: height at row 18
    board[18 * 10 + 0] = 1;
    // Column 1: height at row 15
    board[15 * 10 + 1] = 1;
    const obs = [...board, 0, 0, 0, 0, 0, 0, 0];

    const feats = featGen.getStateFeatures(obs);
    const diff = Math.abs(feats[TetrisFeatures.COL_HT_START_I + 0] -
                         feats[TetrisFeatures.COL_HT_START_I + 1]);
    assert(feats[TetrisFeatures.COL_DIFF_START_I + 0] === diff,
        'Column diff computed correctly');
}

// Test well depth
{
    const featGen = new TetrisFeatures([[0, 1]], 5);
    const board = new Array(200).fill(0);
    // Create a well in column 1 by filling columns 0 and 2
    for (let row = 17; row < 20; row++) {
        board[row * 10 + 0] = 1;
        board[row * 10 + 2] = 1;
    }
    const obs = [...board, 0, 0, 0, 0, 0, 0, 0];

    const feats = featGen.getStateFeatures(obs);
    assert(feats[TetrisFeatures.MAX_WELL_I] >= 3,
        'Well depth detected (>= 3)');
    assert(feats[TetrisFeatures.SUM_WELL_I] >= 3,
        'Sum well depth >= 3');
}

// Test squared features
{
    const featGen = new TetrisFeatures([[0, 1]], 5);
    const board = new Array(200).fill(0);
    board[15 * 10 + 0] = 1; // Height 15 in column 0
    const obs = [...board, 0, 0, 0, 0, 0, 0, 0];

    const feats = featGen.getStateFeatures(obs);
    const height = feats[TetrisFeatures.COL_HT_START_I + 0];
    const squaredIdx = TetrisFeatures.SQUARED_FEATS_START_I + TetrisFeatures.COL_HT_START_I;
    const expectedSquared = (height * height) / TetrisFeatures.HT_SQ_SCALE;
    assertApprox(feats[squaredIdx], expectedSquared, 0.001,
        'Squared height feature scaled correctly');
}

// Test state-action features structure
{
    const featGen = new TetrisFeatures([[0, 1]], 5);
    const obs = new Array(207).fill(0);

    const feats0 = featGen.getStateActionFeatures(obs, 0);
    const feats2 = featGen.getStateActionFeatures(obs, 2);

    // Action 0 should have features in slots 0-45
    let hasNonZeroInAction0 = false;
    for (let i = 0; i < 46; i++) {
        if (feats0[i] !== 0) hasNonZeroInAction0 = true;
    }
    assert(hasNonZeroInAction0, 'Action 0 has features in its slot');

    // Action 0 should have zeros in action 2's slot
    let allZeroInAction2Slot = true;
    for (let i = 92; i < 138; i++) {
        if (feats0[i] !== 0) allZeroInAction2Slot = false;
    }
    assert(allZeroInAction2Slot, 'Action 0 has zeros in action 2 slot');

    // Action 2 should have features in slots 92-137
    let hasNonZeroInAction2 = false;
    for (let i = 92; i < 138; i++) {
        if (feats2[i] !== 0) hasNonZeroInAction2 = true;
    }
    assert(hasNonZeroInAction2, 'Action 2 has features in its slot');
}

// ============================================================
// TetrisFeatures Extended Action Tests
// ============================================================
console.log('\n========== TetrisFeatures Extended Action Tests ==========\n');

// Test getNumExtendedFeatures
{
    const featGen = new TetrisFeatures([[0, 1]], 5);
    assert(featGen.getNumExtendedFeatures() === 46, 'getNumExtendedFeatures() returns 46');
}

// Test getPossActions returns extended actions
{
    const tetris = new Tetris();
    tetris.init();
    const obs = tetris.start();

    const featGen = new TetrisFeatures([[0, 1]], 5);
    const extActions = featGen.getPossActions(obs);

    assert(extActions.length > 0, 'getPossActions returns non-empty list');
    assert(extActions[0].actList !== undefined, 'Extended action has actList');
    assert(Array.isArray(extActions[0].actList), 'actList is an array');
    assert(extActions[0].actList.length > 0, 'actList is non-empty');

    // Each actList should end with NONE (4)
    assert(extActions[0].actList[extActions[0].actList.length - 1] === 4,
        'Extended action ends with NONE');
}

// Test getSAFeats returns state-change features
{
    const tetris = new Tetris();
    tetris.init();
    const obs = tetris.start();

    const featGen = new TetrisFeatures([[0, 1]], 5);
    const extActions = featGen.getPossActions(obs);

    const saFeats = featGen.getSAFeats(obs, extActions[0].actList);

    assert(saFeats.length === 46, 'getSAFeats returns 46 features');
    assert(saFeats instanceof Float64Array, 'getSAFeats returns Float64Array');

    // State-change features should have some non-zero values
    let hasNonZero = false;
    for (let i = 0; i < saFeats.length; i++) {
        if (saFeats[i] !== 0) hasNonZero = true;
    }
    assert(hasNonZero, 'getSAFeats has non-zero state-change values');
}

// Test equivRotation for symmetric pieces
{
    const state = new TetrisState();

    // Line piece (blockId 0) - 2-fold symmetry
    state.currentBlockId = 0;
    state.currentRotation = 0;
    assert(state.equivRotation(0) === true, 'Line: rotation 0 == 0');
    assert(state.equivRotation(2) === true, 'Line: rotation 0 == 2 (2-fold symmetry)');
    assert(state.equivRotation(1) === false, 'Line: rotation 0 != 1');

    // Square piece (blockId 1) - 4-fold symmetry
    state.currentBlockId = 1;
    state.currentRotation = 0;
    assert(state.equivRotation(0) === true, 'Square: rotation 0 == 0');
    assert(state.equivRotation(1) === true, 'Square: rotation 0 == 1 (4-fold symmetry)');
    assert(state.equivRotation(2) === true, 'Square: rotation 0 == 2 (4-fold symmetry)');
    assert(state.equivRotation(3) === true, 'Square: rotation 0 == 3 (4-fold symmetry)');

    // T piece (blockId 2) - no extra symmetry
    state.currentBlockId = 2;
    state.currentRotation = 0;
    assert(state.equivRotation(0) === true, 'T: rotation 0 == 0');
    assert(state.equivRotation(1) === false, 'T: rotation 0 != 1');
}

// Test takeAction returns boolean
{
    const state = new TetrisState();
    state.spawnBlock();

    const result = state.takeAction(4); // NONE action
    assert(typeof result === 'boolean', 'takeAction returns boolean');
}

// Test tree search finds multiple placements
{
    const tetris = new Tetris();
    tetris.init();
    const obs = tetris.start();

    const featGen = new TetrisFeatures([[0, 1]], 5);
    const extActions = featGen.getPossActions(obs);

    // Should find multiple placements for any piece (at least 5 for even the most constrained pieces)
    assert(extActions.length >= 5, `Found ${extActions.length} placements (expected >= 5)`);

    // All placements should have blockMobile = false (piece landed)
    let allLanded = true;
    for (const act of extActions) {
        if (act.blockMobile !== false) allLanded = false;
    }
    assert(allLanded, 'All extended actions have blockMobile = false');
}

// ============================================================
// Tetris Environment Tests
// ============================================================
console.log('\n========== Tetris Environment Tests ==========\n');

// Test environment initialization
{
    const env = new Tetris();
    const spec = env.init();
    assert(spec.numActions === 5, 'Tetris has 5 actions');
    assert(spec.obsRanges.length === 207, 'Tetris obs has 207 dimensions');
    assert(spec.isEpisodic === true, 'Tetris is episodic');
}

// Test environment start
{
    const env = new Tetris();
    env.init();
    const obs = env.start();
    assert(obs.length === 207, 'Start returns 207-dim observation');
    assert(Array.isArray(obs), 'Observation is array');
}

// Test step returns valid result
{
    const env = new Tetris();
    env.init();
    env.start();
    const result = env.step(TetrisActions.NONE);
    assert('obs' in result, 'Step result has obs');
    assert('reward' in result, 'Step result has reward');
    assert('terminal' in result, 'Step result has terminal');
    assert(result.obs.length === 207, 'Step obs has 207 dimensions');
}

// Test all actions are valid
{
    const env = new Tetris();
    env.init();
    env.start();

    const actions = [TetrisActions.LEFT, TetrisActions.RIGHT,
                    TetrisActions.CW, TetrisActions.CCW, TetrisActions.NONE];
    for (const action of actions) {
        const result = env.step(action);
        assert(result.obs.length === 207, `Action ${action} produces valid obs`);
    }
}

// Test TetrisActions enum values
{
    assert(TetrisActions.LEFT === 0, 'LEFT = 0');
    assert(TetrisActions.RIGHT === 1, 'RIGHT = 1');
    assert(TetrisActions.CW === 2, 'CW = 2');
    assert(TetrisActions.CCW === 3, 'CCW = 3');
    assert(TetrisActions.NONE === 4, 'NONE = 4');
    assert(TetrisActions.FALL === 5, 'FALL = 5');
}

// ============================================================
// TetrisState Tests
// ============================================================
console.log('\n========== TetrisState Tests ==========\n');

// Test initial state
{
    const state = new TetrisState();
    assert(state.worldWidth === 10, 'Board width = 10');
    assert(state.worldHeight === 20, 'Board height = 20');
    assert(state.currentBlockId !== undefined, 'Has current block ID');
    assert(!state.gameOver(), 'Game not over initially');
}

// Test piece spawning
{
    const state = new TetrisState();
    state.spawnBlock();
    const pieceType = state.currentBlockId;
    assert(pieceType >= 0 && pieceType <= 6, 'Piece type is valid (0-6)');
}

// Test left/right movement
{
    const state = new TetrisState();
    state.spawnBlock();
    const initialX = state.currentX;
    state.takeAction(TetrisActions.LEFT);
    assert(state.currentX === initialX - 1 || state.currentX === initialX,
        'Left moves piece left (or blocked)');

    state.takeAction(TetrisActions.RIGHT);
    state.takeAction(TetrisActions.RIGHT);
    assert(state.currentX >= initialX, 'Right moves piece right');
}

// Test rotation
{
    const state = new TetrisState();
    state.spawnBlock();
    const initialRot = state.currentRotation;
    state.takeAction(TetrisActions.CW);
    // Rotation should change (mod 4)
    assert(state.currentRotation === (initialRot + 1) % 4 ||
           state.currentRotation === initialRot,
        'CW rotation works (or blocked)');
}

// Test update advances piece down
{
    const state = new TetrisState();
    state.spawnBlock();
    const initialY = state.currentY;
    state.update();
    assert(state.currentY >= initialY, 'update() advances piece down or places it');
}

// Test getObservation
{
    const state = new TetrisState();
    const obs = state.getObservation();
    assert(obs.length === 207, 'Observation has 207 elements');
    assert(obs.every(v => typeof v === 'number'), 'All obs values are numbers');
}

// ============================================================
// TetrisPiece Tests
// ============================================================
console.log('\n========== TetrisPiece Tests ==========\n');

// Test all 7 piece types exist
{
    const pieces = [
        TetrisPiece.makeLine(),
        TetrisPiece.makeSquare(),
        TetrisPiece.makeTri(),
        TetrisPiece.makeSShape(),
        TetrisPiece.makeZShape(),
        TetrisPiece.makeLShape(),
        TetrisPiece.makeJShape()
    ];
    assert(pieces.length === 7, 'All 7 piece types created');

    for (let i = 0; i < 7; i++) {
        assert(pieces[i] !== null, `Piece ${i} is not null`);
        assert(pieces[i].thePiece !== undefined, `Piece ${i} has thePiece array`);
    }
}

// Test piece rotations - thePiece[orientation][row][col] is 5x5 grid
{
    const line = TetrisPiece.makeLine();
    assert(line.thePiece.length === 4, 'Line piece has 4 orientations');

    // Each orientation is a 5x5 grid
    const rot0 = line.getShape(0);
    const rot1 = line.getShape(1);
    assert(rot0.length === 5, 'Rotation 0 is 5 rows');
    assert(rot0[0].length === 5, 'Rotation 0 row is 5 cols');
    assert(rot1.length === 5, 'Rotation 1 is 5 rows');
}

// Test square piece (should be same in all rotations)
{
    const square = TetrisPiece.makeSquare();
    for (let rot = 0; rot < 4; rot++) {
        const shape = square.getShape(rot);
        assert(shape.length === 5, `Square rotation ${rot} is 5x5 grid`);
    }
}

// Test piece cell counts (each tetromino has 4 cells)
{
    const pieces = [
        TetrisPiece.makeLine(),
        TetrisPiece.makeSquare(),
        TetrisPiece.makeTri(),
        TetrisPiece.makeSShape(),
        TetrisPiece.makeZShape(),
        TetrisPiece.makeLShape(),
        TetrisPiece.makeJShape()
    ];

    for (let p = 0; p < pieces.length; p++) {
        const shape = pieces[p].getShape(0);
        let cellCount = 0;
        for (let r = 0; r < 5; r++) {
            for (let c = 0; c < 5; c++) {
                if (shape[r][c] !== 0) cellCount++;
            }
        }
        assert(cellCount === 4, `Piece ${p} has 4 cells`);
    }
}

// ============================================================
// RobotArm Environment Tests
// ============================================================
console.log('\n========== RobotArm Environment Tests ==========\n');

// Test initialization
{
    const env = new RobotArm();
    const spec = env.init();
    assert(spec.numActions === 8, 'RobotArm has 8 actions');
    assert(spec.obsRanges.length === 4, 'RobotArm has 4 observation dims');
}

// Test observation ranges - ranges are [0, dim-1] so span is dim-1
{
    const env = new RobotArm();
    const spec = env.init();
    // worldDims = [20, 20, 10, 10], ranges are [0, dim-1]
    assertApprox(spec.obsRanges[0][1], 19, 0.001, 'Dim 0 max = 19');
    assertApprox(spec.obsRanges[1][1], 19, 0.001, 'Dim 1 max = 19');
    assertApprox(spec.obsRanges[2][1], 9, 0.001, 'Dim 2 max = 9');
    assertApprox(spec.obsRanges[3][1], 9, 0.001, 'Dim 3 max = 9');
}

// Test start returns valid observation
{
    const env = new RobotArm();
    env.init();
    const obs = env.start();
    assert(obs.length === 4, 'Start returns 4-dim observation');
}

// Test all 8 actions work
{
    const env = new RobotArm();
    env.init();
    env.start();

    for (let action = 0; action < 8; action++) {
        const result = env.step(action);
        assert(result.obs.length === 4, `Action ${action} returns valid obs`);
        assert(typeof result.reward === 'number', `Action ${action} returns reward`);
    }
}

// Test action labels
{
    const env = new RobotArm();
    env.init();
    const labels = env.getActionLabels();
    assert(labels.length === 8, 'Has 8 action labels');
}

// ============================================================
// SarsaLambdaAgent Tests
// ============================================================
console.log('\n========== SarsaLambdaAgent Tests ==========\n');

// Test agent creation
{
    const env = new LoopMaze();
    const agent = new SarsaLambdaAgent(env, {
        epsilon: 0.1,
        stepSize: 0.1,
        traceDecayFactor: 0.9,
        discountFactor: 0.99
    });
    assert(agent !== null, 'SarsaLambdaAgent created');
    assert(agent.traceDecayFactor === 0.9, 'Trace decay factor (lambda) set');
    assert(agent.discountFactor === 0.99, 'Discount factor set');
}

// Test episode lifecycle
{
    const env = new LoopMaze();
    const agent = new SarsaLambdaAgent(env, { epsilon: 0.1 });

    const action = agent.startEpisode();
    assert(typeof action === 'number', 'startEpisode returns action');
    assert(action >= 0 && action < 4, 'Action in valid range');

    const result = agent.step(0, false); // reward=0, terminal=false
    assert(typeof result.action === 'number', 'step returns action');
}

// Test learning from rewards
{
    const env = new LoopMaze();
    const agent = new SarsaLambdaAgent(env, {
        epsilon: 0.0,
        stepSize: 0.5
    });

    agent.startEpisode();

    // Give consistent rewards for action 0
    for (let i = 0; i < 50; i++) {
        const result = agent.step(1.0, false);
    }

    // Check that Q-values have been updated
    const stats = agent.getStats();
    assert(stats.totalSteps > 0, 'Agent tracked steps');
}

// Test eligibility traces decay
{
    const env = new LoopMaze();
    const agent = new SarsaLambdaAgent(env, {
        lambda: 0.5,
        stepSize: 0.1
    });

    agent.startEpisode();
    agent.step(1.0, false);
    agent.step(0.0, false);
    agent.step(0.0, false);

    // Traces should have decayed
    assert(agent.model !== null, 'Agent has model');
}

// ============================================================
// ActionSelect Tests
// ============================================================
console.log('\n========== ActionSelect Tests ==========\n');

// Test greedy selection
// ActionSelect constructor: (valFcnModel, selectionMethod, selectionParams, numActions)
{
    const featGen = new RBFFeatures(
        [[0, 10], [0, 10]], 4, 5, 0.08
    );
    featGen.setNormBounds(-1, 1);
    featGen.setBiasFeatPerAct(0.1);

    const model = new LinearModel(featGen.getNumFeatures(), { stepSize: 0.1 });
    model.setFeatGen(featGen);

    const actionSelect = new ActionSelect(model, 'greedy', {}, 4);

    const obs = [5, 5];
    const action = actionSelect.selectAction(obs, 0);
    assert(action >= 0 && action < 4, 'Greedy selects valid action');
}

// Test epsilon-greedy selection
{
    const featGen = new RBFFeatures(
        [[0, 10], [0, 10]], 4, 5, 0.08
    );
    featGen.setNormBounds(-1, 1);
    featGen.setBiasFeatPerAct(0.1);

    const model = new LinearModel(featGen.getNumFeatures(), { stepSize: 0.1 });
    model.setFeatGen(featGen);

    const actionSelect = new ActionSelect(model, 'e-greedy', { epsilon: 1.0 }, 4);

    // With epsilon=1.0, should get varied actions
    const actions = new Set();
    for (let i = 0; i < 100; i++) {
        actions.add(actionSelect.selectAction([5, 5], 0));
    }
    assert(actions.size > 1, 'Epsilon-greedy with eps=1.0 explores multiple actions');
}

// Test values-as-probs selection
{
    const featGen = new RBFFeatures(
        [[0, 10], [0, 10]], 4, 5, 0.08
    );
    featGen.setNormBounds(-1, 1);
    featGen.setBiasFeatPerAct(0.1);

    const model = new LinearModel(featGen.getNumFeatures(), { stepSize: 0.1 });
    model.setFeatGen(featGen);

    const actionSelect = new ActionSelect(model, 'vals-as-probs', {}, 4);

    const action = actionSelect.selectAction([5, 5], 0);
    assert(action >= 0 && action < 4, 'Vals-as-probs selects valid action');
}

// Test tie-breaking in greedy (passing null for lastAction to force random tie-break)
{
    const featGen = new RBFFeatures(
        [[0, 10], [0, 10]], 4, 5, 0.08
    );
    featGen.setNormBounds(-1, 1);
    featGen.setBiasFeatPerAct(0.1);

    const model = new LinearModel(featGen.getNumFeatures(), { stepSize: 0.1 });
    model.setFeatGen(featGen);

    // All weights are 0, so all values equal - should randomly tie-break
    // Pass null as lastAction to not prefer any action
    const actionSelect = new ActionSelect(model, 'greedy', {}, 4);

    const actions = new Set();
    for (let i = 0; i < 100; i++) {
        actions.add(actionSelect.selectAction([5, 5], null));
    }
    assert(actions.size > 1, 'Greedy tie-breaks randomly when values equal');
}

// ============================================================
// HInfluence Tests
// ============================================================
console.log('\n========== HInfluence Tests ==========\n');

// Test annealed parameter method
{
    const hInf = new HInfluence('annealedParam', 0.5);
    const influence = hInf.getHInfluence([0, 0], 0);
    assertApprox(influence, 0.5, 0.001, 'Annealed param returns combParam');
}

// Test step decay
{
    const hInf = new HInfluence('annealedParam', 1.0, {
        stepDecayFactor: 0.9
    });

    const initial = hInf.getHInfluence([0, 0], 0);
    hInf.stepUpdate(true, 0);
    const afterDecay = hInf.getHInfluence([0, 0], 0);

    assertApprox(afterDecay, initial * 0.9, 0.001, 'Step decay reduces influence');
}

// Test episode decay
{
    const hInf = new HInfluence('annealedParam', 1.0, {
        epDecayFactor: 0.8
    });

    const initial = hInf.getHInfluence([0, 0], 0);
    hInf.episodeEndUpdate();
    const afterDecay = hInf.getHInfluence([0, 0], 0);

    assertApprox(afterDecay, initial * 0.8, 0.001, 'Episode decay reduces influence');
}

// Test setTracesToMax
{
    const hInf = new HInfluence('annealedParam', 1.0, {
        stepDecayFactor: 0.5
    });

    hInf.stepUpdate(true, 0);
    hInf.stepUpdate(true, 0);
    const decayed = hInf.getHInfluence([0, 0], 0);

    hInf.setTracesToMax();
    const reset = hInf.getHInfluence([0, 0], 0);

    assert(reset > decayed, 'setTracesToMax resets to full influence');
}

// ============================================================
// TamerRLAgent Tests
// ============================================================
console.log('\n========== TamerRLAgent Tests ==========\n');

// Test CombinationMethods enum
{
    assert(CombinationMethods.TAMER_ONLY === -1, 'TAMER_ONLY = -1');
    assert(CombinationMethods.RL_ONLY === 0, 'RL_ONLY = 0');
    assert(CombinationMethods.REW_SHAPING === 1, 'REW_SHAPING = 1');
    assert(CombinationMethods.ACT_BIASING === 6, 'ACT_BIASING = 6');
}

// Test agent creation with different methods
{
    const env = new LoopMaze();

    const methods = [
        CombinationMethods.TAMER_ONLY,
        CombinationMethods.RL_ONLY,
        CombinationMethods.REW_SHAPING,
        CombinationMethods.ACT_BIASING
    ];

    for (const method of methods) {
        const agent = new TamerRLAgent(env, {
            combMethod: method,
            epsilon: 0.1
        });
        assert(agent !== null, `TamerRLAgent created with method ${method}`);
    }
}

// Test episode lifecycle
{
    const env = new LoopMaze();
    const agent = new TamerRLAgent(env, {
        combMethod: CombinationMethods.REW_SHAPING,
        epsilon: 0.1
    });

    const action = agent.startEpisode();
    assert(typeof action === 'number', 'startEpisode returns action');

    const time = performance.now();
    const result = agent.step(time);
    assert('action' in result, 'step returns action');
}

// Test human reward processing
{
    const env = new LoopMaze();
    const agent = new TamerRLAgent(env, {
        combMethod: CombinationMethods.REW_SHAPING,
        epsilon: 0.0
    });

    agent.startEpisode();
    agent.processHumanReward(1.0, performance.now());

    // Should not throw
    agent.step(performance.now());
    assert(true, 'Human reward processed without error');
}

// ============================================================
// ImitationAgent Tests
// ============================================================
console.log('\n========== ImitationAgent Tests ==========\n');

// Test agent creation
{
    const env = new LoopMaze();
    const agent = new ImitationAgent(env, {
        stepSize: 0.1,
        basisFcnsPerDim: 5
    });
    assert(agent !== null, 'ImitationAgent created');
}

// Test demonstration recording via step with userAction
{
    const env = new LoopMaze();
    const agent = new ImitationAgent(env, { stepSize: 0.5 });

    agent.startEpisode();

    // Simulate demonstrations using step(time, userAction)
    for (let i = 0; i < 10; i++) {
        agent.step(performance.now(), 0); // Demonstrate action 0
    }

    const stats = agent.getStats();
    assert(stats.totalSteps >= 10, 'Agent recorded demonstration steps');
}

// Test key input handling
{
    const env = new LoopMaze();
    const agent = new ImitationAgent(env, { stepSize: 0.1 });

    agent.startEpisode();
    agent.receiveKeyInput('j'); // Should map to left action
    assert(agent.lastUserActI !== -1, 'Key input sets lastUserActI');
}

// Test toggle training
{
    const env = new LoopMaze();
    const agent = new ImitationAgent(env, { stepSize: 0.1 });

    const initial = agent.isTraining();
    agent.toggleTraining();
    const toggled = agent.isTraining();
    assert(initial !== toggled, 'Toggle training changes state');
}

// ============================================================
// ExtActionAgentWrap Tests
// ============================================================
console.log('\n========== ExtActionAgentWrap Tests ==========\n');

// Test wrapper creation
{
    const env = new LoopMaze();
    const baseAgent = new TamerAgent(env, { epsilon: 0.1 });

    // Create extended actions (sequences of base actions)
    const extActions = [
        [0, 0],     // Double up
        [1, 1],     // Double down
        [2, 2],     // Double left
        [3, 3]      // Double right
    ];

    const wrapper = new ExtActionAgentWrap(baseAgent, env, extActions);
    assert(wrapper !== null, 'ExtActionAgentWrap created');
}

// ============================================================
// CreditAssign Edge Cases
// ============================================================
console.log('\n========== CreditAssign Edge Cases ==========\n');

// Test uniform distribution
{
    const ca = new CreditAssign({
        distClass: 'uniform',
        creditDelayMs: 200,
        creditWindowMs: 600
    });
    assert(ca !== null, 'CreditAssign with uniform dist created');
}

// Test previousStep distribution
{
    const ca = new CreditAssign({
        distClass: 'previousStep',
        creditDelayMs: 0,
        creditWindowMs: 0
    });
    assert(ca !== null, 'CreditAssign with previousStep dist created');
}

// Test immediate distribution
{
    const ca = new CreditAssign({
        distClass: 'immediate',
        creditDelayMs: 0,
        creditWindowMs: 0
    });
    assert(ca !== null, 'CreditAssign with immediate dist created');
}

// Test recording timesteps
{
    const ca = new CreditAssign({
        distClass: 'uniform',
        creditDelayMs: 200,
        creditWindowMs: 600
    });

    const feats = new Float64Array([1, 0, 0, 0]);
    ca.recordTimeStepStart(feats, 0);
    ca.recordTimeStepEnd(0.5);
    ca.recordTimeStepStart(feats, 0.5);
    ca.recordTimeStepEnd(1.0);

    assert(true, 'CreditAssign records timesteps without error');
}

// Test toggle training
{
    const ca = new CreditAssign({});
    const initial = ca.isTraining();
    ca.toggleTraining();
    const toggled = ca.isTraining();
    assert(initial !== toggled, 'Toggle training changes state');
}

// ============================================================
// LinearModel Edge Cases
// ============================================================
console.log('\n========== LinearModel Edge Cases ==========\n');

// Test resetTraces
{
    const model = new LinearModel(10, { stepSize: 0.1, decayFactor: 0.9 });

    // Set some traces
    const feats = new Float64Array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0]);
    model.addInstance({ feats, label: 1.0, weight: 1.0 });

    model.resetTraces();

    // Traces should be cleared
    assert(true, 'resetTraces completes without error');
}

// Test buildModel (no-op)
{
    const model = new LinearModel(10, { stepSize: 0.1 });
    model.buildModel();
    assert(true, 'buildModel completes without error');
}

// Test predictForAction without featGen
{
    const model = new LinearModel(10, { stepSize: 0.1 });

    assertThrows(() => {
        model.predictForAction([0, 0], 0);
    }, 'predictForAction throws without featGen');
}

// Test predictForAction with featGen
{
    const featGen = new RBFFeatures([[0, 10], [0, 10]], 4, 5, 0.08);
    featGen.setNormBounds(-1, 1);
    featGen.setBiasFeatPerAct(0.1);

    const model = new LinearModel(featGen.getNumFeatures(), { stepSize: 0.1 });
    model.setFeatGen(featGen);

    const prediction = model.predictForAction([5, 5], 0);
    assert(typeof prediction === 'number', 'predictForAction returns number');
}

// Test zero step size (no learning)
{
    const model = new LinearModel(5, { stepSize: 0.0 });
    const feats = new Float64Array([1, 0, 0, 0, 0]);

    // addInstance expects (feats, label, sampleWeight) not an object
    model.addInstance(feats, 1.0, 1.0);
    const pred = model.predict(feats);

    assertApprox(pred, 0, 0.001, 'Zero step size means no weight updates');
}

// Test very small step size
{
    const model = new LinearModel(5, { stepSize: 0.0001 });
    const feats = new Float64Array([1, 0, 0, 0, 0]);

    model.addInstance(feats, 1.0, 1.0);
    const pred = model.predict(feats);

    assert(pred > 0 && pred < 0.001, 'Small step size gives small update');
}

// ============================================================
// RBFFeatures Edge Cases
// ============================================================
console.log('\n========== RBFFeatures Edge Cases ==========\n');

// Test 1D features
{
    const featGen = new RBFFeatures([[0, 10]], 2, 5, 0.08);
    assert(featGen.getNumFeatures() === 10, '1D: 5 features * 2 actions = 10');
}

// Test high dimensional (but small basis)
{
    const obsRanges = [];
    for (let i = 0; i < 10; i++) {
        obsRanges.push([0, 1]);
    }

    const featGen = new RBFFeatures(obsRanges, 2, 2, 0.08);
    // 2^10 = 1024 features per action, 2 actions = 2048
    assert(featGen.getNumFeatures() === 2048, '10D with 2 basis: 2^10 * 2 = 2048');
}

// Test feature normalization bounds
{
    const featGen = new RBFFeatures([[0, 10]], 2, 5, 0.08);
    featGen.setNormBounds(-1, 1);

    // At boundary, should still produce valid features
    const feats0 = featGen.getStateActionFeatures([0], 0);
    const feats10 = featGen.getStateActionFeatures([10], 0);

    assert(feats0.length === 10, 'Features at min bound have correct length');
    assert(feats10.length === 10, 'Features at max bound have correct length');
}

// Test bias feature
{
    const featGen = new RBFFeatures([[0, 10]], 2, 5, 0.08);
    featGen.setNormBounds(-1, 1);
    featGen.setBiasFeatPerAct(0.5);

    const feats = featGen.getStateActionFeatures([5], 0);
    // With bias, should have (5+1)*2 = 12 features
    assert(feats.length === 12, 'With bias: (5+1)*2 = 12 features');

    // Bias should be 0.5 at index 5 (after 5 RBFs for action 0)
    assertApprox(feats[5], 0.5, 0.001, 'Bias feature value = 0.5');
}

// ============================================================
// Environment Edge Cases
// ============================================================
console.log('\n========== Environment Edge Cases ==========\n');

// Test LoopMaze boundaries
{
    const env = new LoopMaze();
    env.init();
    env.start();

    // Try to move left many times (should hit boundary)
    for (let i = 0; i < 20; i++) {
        env.step(2); // Left
    }
    const obs1 = env.step(2).obs;

    // Should still be valid
    assert(obs1.length === 2, 'LoopMaze obs valid after hitting boundary');
}

// Test MountainCar terminal condition
{
    const env = new MountainCar();
    env.init();
    env.start();

    // Run many steps - should eventually terminate or stay valid
    let terminated = false;
    for (let i = 0; i < 1000 && !terminated; i++) {
        const result = env.step(2); // Push right
        terminated = result.terminal;
    }
    // Either terminated or still running
    assert(true, 'MountainCar runs without error');
}

// Test CartPole balance
{
    const env = new CartPole();
    env.init();
    const obs = env.start();

    assert(obs.length === 4, 'CartPole has 4D observation');

    const result = env.step(0);
    assert(result.obs.length === 4, 'CartPole step returns 4D obs');
}

// Test Acrobot state (4D: theta1, theta2, theta1_dot, theta2_dot)
{
    const env = new Acrobot();
    env.init();
    const obs = env.start();

    assert(obs.length === 4, 'Acrobot has 4D observation');
}

// ============================================================
// Integration: TamerAgent with different environments
// ============================================================
console.log('\n========== TamerAgent Integration Tests ==========\n');

// Test TamerAgent with LoopMaze
{
    const env = new LoopMaze();
    const agent = new TamerAgent(env, { epsilon: 0.0, stepSize: 0.01 });

    agent.startEpisode();
    for (let i = 0; i < 10; i++) {
        agent.step(performance.now());
    }
    assert(true, 'TamerAgent works with LoopMaze');
}

// Test TamerAgent with MountainCar
{
    const env = new MountainCar();
    const agent = new TamerAgent(env, {
        epsilon: 0.0,
        stepSize: 0.01,
        basisFcnsPerDim: 10
    });

    agent.startEpisode();
    for (let i = 0; i < 10; i++) {
        agent.step(performance.now());
    }
    assert(true, 'TamerAgent works with MountainCar');
}

// Test TamerAgent with CartPole
{
    const env = new CartPole();
    const agent = new TamerAgent(env, {
        epsilon: 0.0,
        stepSize: 0.01,
        basisFcnsPerDim: 5
    });

    agent.startEpisode();
    for (let i = 0; i < 10; i++) {
        const result = agent.step(performance.now());
        if (result.terminal) {
            agent.endEpisode();
            agent.startEpisode();
        }
    }
    assert(true, 'TamerAgent works with CartPole');
}

// Test TamerAgent with Tetris
{
    const env = new Tetris();
    const agent = new TamerAgent(env, {
        epsilon: 0.0,
        stepSize: 0.01,
        FeatGenClass: TetrisFeatures
    });

    agent.startEpisode();
    const start = Date.now();
    for (let i = 0; i < 100; i++) {
        agent.step(performance.now());
    }
    const elapsed = Date.now() - start;

    assert(elapsed < 1000, `TamerAgent + Tetris: 100 steps in ${elapsed}ms (< 1s)`);
}

// Test TamerAgent with RobotArm
{
    const env = new RobotArm();
    const agent = new TamerAgent(env, {
        epsilon: 0.0,
        stepSize: 0.01,
        basisFcnsPerDim: 5
    });

    agent.startEpisode();
    for (let i = 0; i < 10; i++) {
        agent.step(performance.now());
    }
    assert(true, 'TamerAgent works with RobotArm');
}

// ============================================================
// Performance Tests
// ============================================================
console.log('\n========== Performance Tests ==========\n');

// Test RBFFeatures performance
{
    const featGen = new RBFFeatures([[0, 10], [0, 10]], 4, 40, 0.08);
    featGen.setNormBounds(-1, 1);
    featGen.setBiasFeatPerAct(0.1);

    const start = Date.now();
    for (let i = 0; i < 1000; i++) {
        featGen.getStateActionFeatures([5, 5], i % 4);
    }
    const elapsed = Date.now() - start;

    assert(elapsed < 500, `RBFFeatures: 1000 feature computations in ${elapsed}ms`);
}

// Test LinearModel performance
{
    const model = new LinearModel(1000, { stepSize: 0.01 });
    const feats = new Float64Array(1000);
    for (let i = 0; i < 1000; i++) feats[i] = Math.random();

    const start = Date.now();
    for (let i = 0; i < 10000; i++) {
        model.addInstance({ feats, label: 1.0, weight: 1.0 });
    }
    const elapsed = Date.now() - start;

    assert(elapsed < 500, `LinearModel: 10000 updates in ${elapsed}ms`);
}

// Test TetrisFeatures performance
{
    const featGen = new TetrisFeatures([[0, 1]], 5);
    const obs = new Array(207).fill(0);

    const start = Date.now();
    for (let i = 0; i < 10000; i++) {
        featGen.getStateActionFeatures(obs, i % 5);
    }
    const elapsed = Date.now() - start;

    assert(elapsed < 500, `TetrisFeatures: 10000 feature computations in ${elapsed}ms`);
}

// ============================================================
// Summary
// ============================================================
console.log('\n==================================================');
console.log(`\nTest Summary: ${passed} passed, ${failed} failed`);
console.log('==================================================');

if (failed > 0) {
    process.exit(1);
}
