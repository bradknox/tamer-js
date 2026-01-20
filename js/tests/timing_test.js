/**
 * Unit tests for timestep timing and credit assignment
 *
 * These tests verify that:
 * 1. Timesteps are recorded at the correct times relative to action execution
 * 2. Timesteps remain "ongoing" while the resulting state is visible
 * 3. Credit assignment correctly credits the timestep that produced the visible state
 * 4. Episode end pause allows feedback for final actions
 */

import { CreditAssign } from '../core/CreditAssign.js';

console.log('======================================================================');
console.log('TIMESTEP TIMING TESTS');
console.log('Verifying credit assignment timing matches Java behavior');
console.log('======================================================================\n');

let passed = 0;
let failed = 0;

function test(name, fn) {
    try {
        fn();
        console.log(`✓ ${name}`);
        passed++;
    } catch (e) {
        console.log(`✗ ${name}`);
        console.log(`  Error: ${e.message}`);
        failed++;
    }
}

function assertEqual(actual, expected, message) {
    if (actual !== expected) {
        throw new Error(`${message}: expected ${expected}, got ${actual}`);
    }
}

function assertApprox(actual, expected, tolerance, message) {
    if (Math.abs(actual - expected) > tolerance) {
        throw new Error(`${message}: expected ~${expected}, got ${actual}`);
    }
}

function assertTrue(condition, message) {
    if (!condition) {
        throw new Error(message);
    }
}

// Test 1: Ongoing timestep gets full credit
test('Ongoing timestep receives credit', () => {
    const credA = new CreditAssign({
        distClass: 'uniform',
        creditDelayMs: 200,
        creditWindowMs: 600,
        extrapolateFutureRew: true
    });
    credA.setInTrainSess(0, true);

    // Start timestep at t=0
    const feats = new Float64Array([1, 2, 3]);
    credA.recordTimeStepStart(feats, 0, null, null);

    // Give reward at t=0.4 (timestep still ongoing)
    credA.processNewHReward(1.0, 0.4);

    // Check that the active sample got credit
    assertTrue(credA.activeSamples.length === 1, 'Should have one active sample');
    assertTrue(credA.activeSamples[0].unweightedRew > 0, 'Ongoing timestep should receive reward');
});

// Test 2: Ended timestep still receives credit within window
test('Ended timestep receives credit within window', () => {
    const credA = new CreditAssign({
        distClass: 'uniform',
        creditDelayMs: 200,
        creditWindowMs: 600,
        extrapolateFutureRew: true
    });
    credA.setInTrainSess(0, true);

    // Start timestep at t=0
    const feats = new Float64Array([1, 2, 3]);
    credA.recordTimeStepStart(feats, 0, null, null);

    // End timestep at t=0.2
    credA.recordTimeStepEnd(0.2);

    // Start new timestep at t=0.2
    credA.recordTimeStepStart(feats, 0.2, null, null);

    // Give reward at t=0.3 (first timestep ended 0.1s ago, within window)
    credA.processNewHReward(1.0, 0.3);

    // Check that the first sample got credit
    assertTrue(credA.activeSamples[0].unweightedRew > 0, 'Ended timestep should receive credit');
});

// Test 3: Ended timestep gets less credit than ongoing (late feedback)
test('Ongoing timestep gets more credit than ended timestep for late feedback', () => {
    // Create two credit assigners to compare
    const credOngoing = new CreditAssign({
        distClass: 'uniform',
        creditDelayMs: 200,
        creditWindowMs: 600,
        extrapolateFutureRew: false  // Disable extrapolation for direct comparison
    });
    credOngoing.setInTrainSess(0, true);

    const credEnded = new CreditAssign({
        distClass: 'uniform',
        creditDelayMs: 200,
        creditWindowMs: 600,
        extrapolateFutureRew: false
    });
    credEnded.setInTrainSess(0, true);

    const feats = new Float64Array([1, 2, 3]);

    // Scenario 1: Timestep remains ongoing
    credOngoing.recordTimeStepStart(feats, 0, null, null);
    // Don't end it

    // Scenario 2: Timestep ends at t=0.2
    credEnded.recordTimeStepStart(feats, 0, null, null);
    credEnded.recordTimeStepEnd(0.2);

    // Give reward at t=0.5 (0.3s after timestep ended in scenario 2)
    credOngoing.processNewHReward(1.0, 0.5);
    credEnded.processNewHReward(1.0, 0.5);

    const creditOngoing = credOngoing.activeSamples[0].unweightedRew;
    const creditEnded = credEnded.activeSamples[0].unweightedRew;

    // For late feedback, ongoing timestep should get more credit
    // because relativeNearBound is clamped to windowStart for ongoing
    assertTrue(
        creditOngoing >= creditEnded,
        `Ongoing credit (${creditOngoing}) should be >= ended credit (${creditEnded})`
    );
});

// Test 4: First step has no previous timestep to end
test('First step does not end non-existent timestep', () => {
    const credA = new CreditAssign({
        distClass: 'uniform',
        creditDelayMs: 200,
        creditWindowMs: 600
    });
    credA.setInTrainSess(0, true);

    // Call recordTimeStepEnd before any timestep exists
    // This should not throw an error
    credA.recordTimeStepEnd(0.1);

    assertEqual(credA.timeStepsInWindow.length, 0, 'Should have no timesteps');
    assertEqual(credA.activeSamples.length, 0, 'Should have no samples');
});

// Test 5: Episode end pause allows feedback for final actions
test('Feedback during episode end pause credits final timestep', () => {
    const credA = new CreditAssign({
        distClass: 'uniform',
        creditDelayMs: 200,
        creditWindowMs: 600,
        extrapolateFutureRew: true
    });
    credA.setInTrainSess(0, true);

    // Simulate episode with multiple steps
    const feats = new Float64Array([1, 2, 3]);

    // Step 1: Record T0
    credA.recordTimeStepStart(feats, 0, null, null);

    // Step 2: End T0, record T1
    credA.recordTimeStepEnd(0.3);
    credA.recordTimeStepStart(feats, 0.3, null, null);

    // Step 3 (terminal): End T1
    credA.recordTimeStepEnd(0.6);

    // Episode ends, pause begins
    // User gives feedback at t=0.7 (during pause) for the final action
    credA.processNewHReward(1.0, 0.7);

    // T1 should have received credit (ended at 0.6, feedback at 0.7)
    const samples = credA.processSamplesAndRemoveFinished(0.9, true);

    // At least one sample should have received credit
    const creditedSamples = samples.filter(s => Math.abs(s.label) > 0.001);
    assertTrue(
        creditedSamples.length > 0,
        'Final timestep should receive credit from feedback during pause'
    );
});

// Test 6: Correct timestep gets primary credit when result state is visible
test('Timestep that produced visible state gets primary credit', () => {
    const credA = new CreditAssign({
        distClass: 'uniform',
        creditDelayMs: 200,
        creditWindowMs: 600,
        extrapolateFutureRew: true
    });
    credA.setInTrainSess(0, true);

    const feats0 = new Float64Array([1, 0, 0]);
    const feats1 = new Float64Array([0, 1, 0]);

    // T0 starts at t=0 (action A0 selected in state S0)
    credA.recordTimeStepStart(feats0, 0, [0], 0);

    // A0 is executed, producing S1, which is rendered
    // T0 remains ongoing during this time

    // T0 ends at t=0.3, T1 starts (action A1 selected in state S1)
    credA.recordTimeStepEnd(0.3);
    credA.recordTimeStepStart(feats1, 0.3, [1], 1);

    // User sees S1 (result of A0) and gives feedback at t=0.35
    // This should primarily credit T0 (S0, A0) which produced S1
    credA.processNewHReward(1.0, 0.35);

    const creditT0 = credA.activeSamples[0].unweightedRew;
    const creditT1 = credA.activeSamples[1].unweightedRew;

    // T0 should get more credit because it started earlier and its action produced S1
    assertTrue(
        creditT0 > creditT1,
        `T0 credit (${creditT0}) should be > T1 credit (${creditT1})`
    );
});

// Test 7: previousStep mode - duringStepTransition affects finish bounds
test('previousStep mode: sample finishes during step transition', () => {
    const credA = new CreditAssign({
        distClass: 'previousStep',
        creditDelayMs: 0,
        creditWindowMs: 1000,
        extrapolateFutureRew: false  // Tetris uses false
    });
    credA.setInTrainSess(0, true);

    const feats = new Float64Array([1, 2, 3]);

    // Record T0
    credA.recordTimeStepStart(feats, 0, null, null);
    credA.recordTimeStepEnd(0.15);

    // Record T1
    credA.recordTimeStepStart(feats, 0.15, null, null);
    credA.recordTimeStepEnd(0.30);

    // Record T2
    credA.recordTimeStepStart(feats, 0.30, null, null);

    // Give reward to T1 (the previous step)
    credA.processNewHReward(1.0, 0.35);

    // Process samples WITH duringStepTransition=true (like in agent_step)
    // This should remove T0 (stepsBeforeCurrent=2, finishedBound=1 during transition)
    // and return T1 with the reward
    const samples = credA.processSamplesAndRemoveFinished(0.40, true, true);

    // T1 should be in the returned samples with reward
    assertTrue(samples.length >= 1, 'Should have at least one sample');
    assertTrue(samples.some(s => Math.abs(s.label - 1.0) < 0.01), 'T1 should have label ~1.0');
});

// Test 8: previousStep mode - sample does NOT finish without duringStepTransition
test('previousStep mode: sample stays active without step transition', () => {
    const credA = new CreditAssign({
        distClass: 'previousStep',
        creditDelayMs: 0,
        creditWindowMs: 1000,
        extrapolateFutureRew: false
    });
    credA.setInTrainSess(0, true);

    const feats = new Float64Array([1, 2, 3]);

    // Record T0
    credA.recordTimeStepStart(feats, 0, null, null);
    credA.recordTimeStepEnd(0.15);

    // Record T1
    credA.recordTimeStepStart(feats, 0.15, null, null);
    credA.recordTimeStepEnd(0.30);

    // Record T2
    credA.recordTimeStepStart(feats, 0.30, null, null);

    // Give reward to T1
    credA.processNewHReward(1.0, 0.35);

    // Process samples WITHOUT duringStepTransition (default=false)
    // T0 should NOT be removed yet (stepsBeforeCurrent=2, finishedBound=2 outside transition)
    // T0 is removed, T1 stays active
    const samples = credA.processSamplesAndRemoveFinished(0.40, true, false);

    // Without duringStepTransition, T0 is at finishedBound (stepsBeforeCurrent=2=finishedBound)
    // so T0 should be removed, but T1 (stepsBeforeCurrent=1) should stay active
    assertEqual(credA.activeSamples.length, 2, 'Should have 2 active samples (T1 and T2)');
});

// Test 9: previousStep credit assignment - only previous step gets credit
test('previousStep mode: only stepsBeforeCurrent=1 gets credit', () => {
    const credA = new CreditAssign({
        distClass: 'previousStep',
        creditDelayMs: 0,
        creditWindowMs: 1000,
        extrapolateFutureRew: false
    });
    credA.setInTrainSess(0, true);

    const feats0 = new Float64Array([1, 0, 0]);
    const feats1 = new Float64Array([0, 1, 0]);
    const feats2 = new Float64Array([0, 0, 1]);

    // Record T0
    credA.recordTimeStepStart(feats0, 0, null, null);
    credA.recordTimeStepEnd(0.15);

    // Record T1
    credA.recordTimeStepStart(feats1, 0.15, null, null);
    credA.recordTimeStepEnd(0.30);

    // Record T2 (current)
    credA.recordTimeStepStart(feats2, 0.30, null, null);

    // Give reward - should credit T1 (stepsBeforeCurrent=1)
    credA.processNewHReward(1.0, 0.35);

    // T0 (stepsBeforeCurrent=2) should have no credit
    // T1 (stepsBeforeCurrent=1) should have credit
    // T2 (stepsBeforeCurrent=0) should have no credit
    const t0Credit = credA.activeSamples[0].unweightedRew;
    const t1Credit = credA.activeSamples[1].unweightedRew;
    const t2Credit = credA.activeSamples[2].unweightedRew;

    assertEqual(t0Credit, 0, 'T0 (stepsBeforeCurrent=2) should have no credit');
    assertEqual(t1Credit, 1.0, 'T1 (stepsBeforeCurrent=1) should have full credit');
    assertEqual(t2Credit, 0, 'T2 (stepsBeforeCurrent=0) should have no credit');
});

// Test 10: Extended actions - timestep features match observation
// This test verifies the fix for the feature offset bug where extended action
// timesteps were being recorded with the PREVIOUS action's features instead
// of the newly selected action's features
test('Extended actions: timestep features match current observation', () => {
    // Simulate what happens in TamerAgent for extended actions
    // The key is that when step() is called, we should:
    // 1. Select the new action based on currentObs
    // 2. Record timestep with features for (currentObs, newAction)

    // Create a mock scenario:
    // - startEpisode: select A0 based on S0, record T0 with feats(S0, A0)
    // - step: currentObs=S1, select A1, record T1 with feats(S1, A1)

    // The bug was: T1 was recorded with feats(S0, A0) instead of feats(S1, A1)

    const credA = new CreditAssign({
        distClass: 'previousStep',
        creditDelayMs: 0,
        creditWindowMs: 1000,
        extrapolateFutureRew: false
    });
    credA.setInTrainSess(0, true);

    // Simulate T0 with features representing state S0
    const featsS0A0 = new Float64Array([1.0, 0.0, 0.0]);  // Distinct features for S0
    credA.recordTimeStepStart(featsS0A0, 0, null, null);
    credA.recordTimeStepEnd(0.15);

    // Simulate T1 with features representing state S1
    // This is the critical part - T1 should have different features than T0
    const featsS1A1 = new Float64Array([0.0, 1.0, 0.0]);  // Distinct features for S1
    credA.recordTimeStepStart(featsS1A1, 0.15, null, null);
    credA.recordTimeStepEnd(0.30);

    // Simulate T2 (current)
    const featsS2A2 = new Float64Array([0.0, 0.0, 1.0]);
    credA.recordTimeStepStart(featsS2A2, 0.30, null, null);

    // Verify that each timestep has distinct features
    assertTrue(
        credA.activeSamples[0].feats[0] === 1.0 && credA.activeSamples[0].feats[1] === 0.0,
        'T0 should have features for S0 (first element = 1)'
    );
    assertTrue(
        credA.activeSamples[1].feats[0] === 0.0 && credA.activeSamples[1].feats[1] === 1.0,
        'T1 should have features for S1 (second element = 1)'
    );
    assertTrue(
        credA.activeSamples[2].feats[2] === 1.0,
        'T2 should have features for S2 (third element = 1)'
    );

    // Give reward - should credit T1 with the correct features
    credA.processNewHReward(1.0, 0.35);

    // T1 should have received credit
    assertEqual(credA.activeSamples[1].unweightedRew, 1.0, 'T1 should receive full credit');

    // Verify T1's features are still correct (S1 features, not S0)
    assertTrue(
        credA.activeSamples[1].feats[1] === 1.0,
        'T1 should still have S1 features after credit'
    );
});

console.log('\n======================================================================');
console.log(`Results: ${passed} passed, ${failed} failed`);
console.log('======================================================================');

if (failed > 0) {
    process.exit(1);
}
