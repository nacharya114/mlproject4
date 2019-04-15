package rlDriver;

import java.awt.Color;
import java.util.List;

import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import burlap.behavior.policy.GreedyQPolicy;
import burlap.behavior.policy.Policy;
import burlap.behavior.policy.PolicyUtils;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.auxiliary.StateReachability;
import burlap.behavior.singleagent.auxiliary.performance.LearningAlgorithmExperimenter;
import burlap.behavior.singleagent.auxiliary.performance.PerformanceMetric;
import burlap.behavior.singleagent.auxiliary.performance.TrialMode;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.ValueFunctionVisualizerGUI;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.ArrowActionGlyph;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.LandmarkColorBlendInterpolation;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.PolicyGlyphPainter2D;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.StateValuePainter2D;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.learning.LearningAgentFactory;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.learning.tdmethods.SarsaLam;
import burlap.behavior.singleagent.planning.Planner;
import burlap.behavior.singleagent.planning.deterministic.DeterministicPlanner;
import burlap.behavior.singleagent.planning.deterministic.SDPlannerPolicy;
import burlap.behavior.singleagent.planning.deterministic.informed.Heuristic;
import burlap.behavior.singleagent.planning.deterministic.informed.NullHeuristic;
import burlap.behavior.singleagent.planning.deterministic.informed.astar.AStar;
import burlap.behavior.singleagent.planning.deterministic.uninformed.bfs.BFS;
import burlap.behavior.singleagent.planning.deterministic.uninformed.dfs.DFS;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.behavior.valuefunction.QProvider;
import burlap.behavior.valuefunction.ValueFunction;
import burlap.domain.singleagent.blockdude.BlockDude;
import burlap.domain.singleagent.blockdude.BlockDudeLevelConstructor;
import burlap.domain.singleagent.blockdude.BlockDudeTF;
import  burlap.domain.singleagent.blockdude.BlockDudeVisualizer;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
import burlap.domain.singleagent.gridworld.state.GridAgent;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.mdp.auxiliary.stateconditiontest.StateConditionTest;
import burlap.mdp.auxiliary.stateconditiontest.TFGoalCondition;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.core.state.State;
import burlap.mdp.core.state.vardomain.VariableDomain;
import burlap.mdp.singleagent.SADomain;
import burlap.mdp.singleagent.common.GoalBasedRF;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.model.FactoredModel;
import burlap.statehashing.HashableStateFactory;
import burlap.statehashing.simple.SimpleHashableStateFactory;
import burlap.visualizer.Visualizer;




public class TestBlockDude {
	
	public final static int MAXX = 24;
	public final static int MAXY = 25;
	
	BlockDude constructor;
	SADomain domain;
	TerminalFunction tf;
	HashableStateFactory hashingFactory;
	SimulatedEnvironment env;
	
	State initialState;
	StateConditionTest goalCondition;
	
	public TestBlockDude() {
		constructor = new BlockDude(MAXX, MAXY);
		domain = constructor.generateDomain();
		initialState = BlockDudeLevelConstructor.getLevel1(domain);
		
		tf = constructor.getTf();
		goalCondition = new TFGoalCondition(tf);
		
		hashingFactory = new SimpleHashableStateFactory();

		env = new SimulatedEnvironment(domain, initialState);
	}
	
	public void visualize(String outputpath){
		Visualizer v = BlockDudeVisualizer.getVisualizer(MAXX, MAXY);
		new EpisodeSequenceVisualizer(v, domain, outputpath);
	}
	
	
	public void valueIterationExample(String outputPath){

		Planner planner = new ValueIteration(domain, 0.99, hashingFactory, 0.001, 100);
		Policy p = planner.planFromState(initialState);

		PolicyUtils.rollout(p, initialState, domain.getModel()).write(outputPath + "vi");

		simpleValueFunctionVis((ValueFunction)planner, p);
		//manualValueFunctionVis((ValueFunction)planner, p);

	}


	public void qLearningExample(String outputPath){

		LearningAgent agent = new QLearning(domain, 0.99, hashingFactory, 0., 1.);

		System.out.println("Starting Q-Learning");
		//run learning for 50 episodes
		for(int i = 0; i < 50; i++){
			Episode e = agent.runLearningEpisode(env);

			e.write(outputPath + "ql_" + i);
			System.out.println(i + ": " + e.maxTimeStep());

			//reset environment for next learning episode
			env.resetEnvironment();
		}

		simpleValueFunctionVis((ValueFunction)agent, new GreedyQPolicy((QProvider) agent));

	}


	public void sarsaLearningExample(String outputPath){

		LearningAgent agent = new SarsaLam(domain, 0.99, hashingFactory, 0., 0.5, 0.3);

		//run learning for 50 episodes
		for(int i = 0; i < 50; i++){
			Episode e = agent.runLearningEpisode(env);

			e.write(outputPath + "sarsa_" + i);
			System.out.println(i + ": " + e.maxTimeStep());

			//reset environment for next learning episode
			env.resetEnvironment();
		}

	}

	public void simpleValueFunctionVis(ValueFunction valueFunction, Policy p){

		List<State> allStates = StateReachability.getReachableStates(initialState, domain, hashingFactory);
		ValueFunctionVisualizerGUI gui = GridWorldDomain.getGridWorldValueFunctionVisualization(allStates, 11, 11, valueFunction, p);
		gui.initGUI();

	}

	public void manualValueFunctionVis(ValueFunction valueFunction, Policy p){

		List<State> allStates = StateReachability.getReachableStates(initialState, domain, hashingFactory);

		//define color function
		LandmarkColorBlendInterpolation rb = new LandmarkColorBlendInterpolation();
		rb.addNextLandMark(0., Color.RED);
		rb.addNextLandMark(1., Color.BLUE);

		//define a 2D painter of state values, specifying which attributes correspond to the x and y coordinates of the canvas
		StateValuePainter2D svp = new StateValuePainter2D(rb);
		svp.setXYKeys("agent:x", "agent:y", new VariableDomain(0, 11), new VariableDomain(0, 11), 1, 1);

		//create our ValueFunctionVisualizer that paints for all states
		//using the ValueFunction source and the state value painter we defined
		ValueFunctionVisualizerGUI gui = new ValueFunctionVisualizerGUI(allStates, svp, valueFunction);

		//define a policy painter that uses arrow glyphs for each of the grid world actions
		PolicyGlyphPainter2D spp = new PolicyGlyphPainter2D();
		spp.setXYKeys("agent:x", "agent:y", new VariableDomain(0, 11), new VariableDomain(0, 11), 1, 1);

		spp.setActionNameGlyphPainter(GridWorldDomain.ACTION_NORTH, new ArrowActionGlyph(0));
		spp.setActionNameGlyphPainter(GridWorldDomain.ACTION_SOUTH, new ArrowActionGlyph(1));
		spp.setActionNameGlyphPainter(GridWorldDomain.ACTION_EAST, new ArrowActionGlyph(2));
		spp.setActionNameGlyphPainter(GridWorldDomain.ACTION_WEST, new ArrowActionGlyph(3));
		spp.setRenderStyle(PolicyGlyphPainter2D.PolicyGlyphRenderStyle.DISTSCALED);


		//add our policy renderer to it
		gui.setSpp(spp);
		gui.setPolicy(p);

		//set the background color for places where states are not rendered to grey
		gui.setBgColor(Color.GRAY);

		//start it
		gui.initGUI();



	}


	public void experimentAndPlotter(){

		//different reward function for more structured performance plots
		((FactoredModel)domain.getModel()).setRf(new GoalBasedRF(this.goalCondition, 5.0, -0.1));

		/**
		 * Create factories for Q-learning agent and SARSA agent to compare
		 */
		LearningAgentFactory qLearningFactory = new LearningAgentFactory() {

			public String getAgentName() {
				return "Q-Learning";
			}


			public LearningAgent generateAgent() {
				return new QLearning(domain, 0.99, hashingFactory, 0.3, 0.1);
			}
		};

		LearningAgentFactory sarsaLearningFactory = new LearningAgentFactory() {

			public String getAgentName() {
				return "SARSA";
			}


			public LearningAgent generateAgent() {
				return new SarsaLam(domain, 0.99, hashingFactory, 0.0, 0.1, 1.);
			}
		};

		LearningAlgorithmExperimenter exp = new LearningAlgorithmExperimenter(env, 10, 100, qLearningFactory, sarsaLearningFactory);
		exp.setUpPlottingConfiguration(500, 250, 2, 1000,
				TrialMode.MOST_RECENT_AND_AVERAGE,
				PerformanceMetric.CUMULATIVE_STEPS_PER_EPISODE,
				PerformanceMetric.AVERAGE_EPISODE_REWARD);

		exp.startExperiment();
		exp.writeStepAndEpisodeDataToCSV("expData");

	}


	public static void main(String[] args) {

		TestBlockDude example = new TestBlockDude();
		String outputPath = "output1/";

//		example.BFSExample(outputPath);
		//example.DFSExample(outputPath);
		//example.AStarExample(outputPath);
//		example.valueIterationExample(outputPath);
//		example.qLearningExample(outputPath);
//		example.sarsaLearningExample(outputPath);

		example.experimentAndPlotter();

//		example.visualize(outputPath);

	}
	
}

//public class TestBlockDude {
//	SADomain domain;
//	BlockDude constructor;
//	
//	@Before
//	public void setup() {
//		constructor = new BlockDude();
//		domain = constructor.generateDomain();
//	}
//
//	@After
//	public void teardown() {
//		this.domain = null;
//		this.constructor = null;
//	}
//	
//	public State generateState() {
//		return BlockDudeLevelConstructor.getLevel3(domain);
//	}
//
//	@Test
//	public void testDude() {
//		State s = this.generateState();
//		this.testDude(s);
//	}
//	
//	public void testDude(State s) {
//		TerminalFunction tf = new BlockDudeTF();
//		StateConditionTest sc = new TFGoalCondition(tf);
//
//		AStar astar = new AStar(domain, sc, new SimpleHashableStateFactory(), new NullHeuristic());
//		astar.toggleDebugPrinting(false);
//		astar.planFromState(s);
//
//		Policy p = new SDPlannerPolicy(astar);
//		Episode ea = PolicyUtils.rollout(p, s, domain.getModel(), 100);
//
//		State lastState = ea.stateSequence.get(ea.stateSequence.size() - 1);
//		Assert.assertEquals(true, tf.isTerminal(lastState));
//		Assert.assertEquals(true, sc.satisfies(lastState));
//		Assert.assertEquals(-94.0, ea.discountedReturn(1.0), 0.001);
//
//		/*
//		BlockDude constructor = new BlockDude();
//		Domain d = constructor.generateDomain();
//		List<Integer> px = new ArrayList<Integer>();
//		List <Integer> ph = new ArrayList<Integer>();
//		ph.add(15);
//		ph.add(3);
//		ph.add(3);
//		ph.add(3);
//		ph.add(0);
//		ph.add(0);
//		ph.add(0);
//		ph.add(1);
//		ph.add(2);
//		ph.add(0);
//		ph.add(2);
//		ph.add(3);
//		ph.add(2);
//		ph.add(2);
//		ph.add(3);
//		ph.add(3);
//		ph.add(15);
//		
//		State o = BlockDude.getCleanState(d, px, ph, 6);
//		o = BlockDude.setAgent(o, 9, 3, 1, 0);
//		o = BlockDude.setExit(o, 1, 0);
//		
//		o = BlockDude.setBlock(o, 0, 5, 1);
//		o = BlockDude.setBlock(o, 1, 6, 1);
//		o = BlockDude.setBlock(o, 2, 14, 3);
//		o = BlockDude.setBlock(o, 3, 16, 4);
//		o = BlockDude.setBlock(o, 4, 17, 4);
//		o = BlockDude.setBlock(o, 5, 17, 5);
//		
//		TerminalFunction tf = new SinglePFTF(d.getPropFunction(BlockDude.PFATEXIT));
//		StateConditionTest sc = new SinglePFSCT(d.getPropFunction(BlockDude.PFATEXIT));
//		RewardFunction rf = new UniformCostRF();
//		AStar astar = new AStar(d, rf, sc, new DiscreteStateHashFactory(), new NullHeuristic());
//		astar.toggleDebugPrinting(false);
//		astar.planFromState(o);
//		Policy p = new SDPlannerPolicy(astar);
//		EpisodeAnalysis ea = p.evaluateBehavior(o, rf, tf, 100);
//		State lastState = ea.stateSequence.get(ea.stateSequence.size() - 1);
//		Assert.assertEquals(true, tf.isTerminal(lastState));
//		Assert.assertEquals(true, sc.satisfies(lastState));
//		Assert.assertEquals(-94.0, ea.getDiscountedReturn(1.0), 0.001);
//		*/
//	}
//}
