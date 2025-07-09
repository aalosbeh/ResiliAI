"""
Experiment Runner for ResiliAI Framework

This module orchestrates experiments using the ResiliAI framework components,
collects results, and generates visualizations for analysis.

Author: Anas ALsobeh, Raneem Alkurdi
Date: July 2025
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm

# Import ResiliAI modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_preprocessing import DataPreprocessor
from src.digital_twin import DigitalTwinEconomy
from src.resilience_scoring import ResilienceScoringEngine, generate_synthetic_data
from src.marl_framework import env as marl_env


class ExperimentRunner:
    """
    Orchestrates experiments using the ResiliAI framework components,
    collects results, and generates visualizations.
    """
    
    def __init__(self, config=None):
        """
        Initialize the experiment runner with configuration.
        
        Args:
            config (dict, optional): Configuration parameters for experiments
        """
        self.config = config or {
            'data_dir': '/home/ubuntu/ESI2025_ResiliAI/data',
            'results_dir': '/home/ubuntu/ESI2025_ResiliAI/results',
            'visualizations_dir': '/home/ubuntu/ESI2025_ResiliAI/results/visualizations',
            'random_seed': 42,
            'experiment_scenarios': ['pandemic', 'financial_crisis', 'climate_shock'],
            'policy_interventions': ['fiscal_stimulus', 'monetary_easing', 'targeted_support'],
            'simulation_steps': 60,
            'num_trials': 5
        }
        
        # Create directories if they don't exist
        os.makedirs(self.config['data_dir'], exist_ok=True)
        os.makedirs(self.config['results_dir'], exist_ok=True)
        os.makedirs(self.config['visualizations_dir'], exist_ok=True)
        
        # Set random seed for reproducibility
        np.random.seed(self.config['random_seed'])
        
        # Initialize components
        self.data_preprocessor = None
        self.digital_twin = None
        self.resilience_engine = None
        self.marl_environment = None
        
        # Store experiment results
        self.results = {}
        
    def prepare_data(self):
        """
        Prepare data for experiments using the DataPreprocessor.
        
        Returns:
            dict: Processed data for experiments
        """
        print("Preparing data for experiments...")
        
        # Initialize data preprocessor
        self.data_preprocessor = DataPreprocessor(self.config['data_dir'])
        
        # Check if IMF fiscal policy data exists
        imf_text_path = os.path.join(self.config['data_dir'], 'imf_fiscal_policy_covid19.txt')
        if os.path.exists(imf_text_path):
            # Process real IMF data
            print("Processing IMF fiscal policy data...")
            self.data_preprocessor.load_text_data("imf_fiscal_policy_covid19.txt", "imf_fiscal")
            self.data_preprocessor.extract_country_data("imf_fiscal")
            self.data_preprocessor.extract_fiscal_measures("imf_fiscal")
            self.data_preprocessor.clean_data("imf_fiscal")
            self.data_preprocessor.normalize_data("imf_fiscal")
            self.data_preprocessor.engineer_features("imf_fiscal")
            
            # Save processed data
            processed_path = os.path.join(self.config['data_dir'], 'processed_fiscal_data.csv')
            self.data_preprocessor.save_processed_data("imf_fiscal", processed_path)
            
            # Prepare data for simulation
            real_data = self.data_preprocessor.prepare_for_simulation("imf_fiscal")
            print(f"Prepared real data for {len(real_data)} entities")
            
            return {'real_data': real_data}
        else:
            # Generate synthetic data if real data is not available
            print("Real IMF data not found. Generating synthetic data...")
            synthetic_df = generate_synthetic_data(n_samples=30, n_features=12)
            
            # Save synthetic data
            synthetic_path = os.path.join(self.config['data_dir'], 'synthetic_economic_data.csv')
            synthetic_df.to_csv(synthetic_path, index=False)
            
            # Convert to simulation format
            synthetic_data = {}
            for _, row in synthetic_df.iterrows():
                entity = row['entity']
                entity_data = row.drop(['entity']).to_dict()
                synthetic_data[entity] = entity_data
            
            print(f"Generated synthetic data for {len(synthetic_data)} entities")
            
            return {'synthetic_data': synthetic_data}
    
    def run_digital_twin_experiments(self, initial_data):
        """
        Run experiments using the Digital Twin Economy simulation.
        
        Args:
            initial_data (dict): Initial economic data for simulation
            
        Returns:
            dict: Experiment results
        """
        print("Running Digital Twin Economy experiments...")
        
        # Initialize digital twin
        self.digital_twin = DigitalTwinEconomy(initial_data, {
            'simulation_steps': self.config['simulation_steps'],
            'random_seed': self.config['random_seed']
        })
        
        experiment_results = {}
        
        # Run baseline simulation (no shocks or policies)
        print("Running baseline simulation...")
        baseline_summary = self.digital_twin.run_simulation(steps=self.config['simulation_steps'])
        experiment_results['baseline'] = baseline_summary
        
        # Run shock scenarios
        for scenario in self.config['experiment_scenarios']:
            print(f"Running {scenario} shock scenario...")
            
            # Reset digital twin for this scenario
            self.digital_twin = DigitalTwinEconomy(initial_data, {
                'simulation_steps': self.config['simulation_steps'],
                'random_seed': self.config['random_seed']
            })
            
            # Apply shock at step 10
            self.digital_twin.simulate_step(apply_random_shocks=False)  # Step 1
            shock_impact = self.digital_twin.apply_shock(
                shock_type=scenario.replace('_', ' '),
                severity=0.7,
                affected_entities=None  # All entities
            )
            
            # Continue simulation
            scenario_summary = self.digital_twin.run_simulation(steps=self.config['simulation_steps']-1)
            scenario_summary['shock_impact'] = shock_impact
            
            experiment_results[f'shock_{scenario}'] = scenario_summary
        
        # Run policy response experiments
        for scenario in self.config['experiment_scenarios']:
            for policy in self.config['policy_interventions']:
                print(f"Running {policy} response to {scenario} shock...")
                
                # Reset digital twin for this scenario
                self.digital_twin = DigitalTwinEconomy(initial_data, {
                    'simulation_steps': self.config['simulation_steps'],
                    'random_seed': self.config['random_seed']
                })
                
                # Apply shock at step 10
                for i in range(10):
                    self.digital_twin.simulate_step(apply_random_shocks=False)
                
                shock_impact = self.digital_twin.apply_shock(
                    shock_type=scenario.replace('_', ' '),
                    severity=0.7,
                    affected_entities=None  # All entities
                )
                
                # Apply policy response at step 15
                for i in range(5):
                    self.digital_twin.simulate_step(apply_random_shocks=False)
                
                policy_effect = self.digital_twin.apply_policy(
                    policy_type=policy,
                    target_entities=None,  # All entities
                    intensity=0.8
                )
                
                # Continue simulation
                response_summary = self.digital_twin.run_simulation(steps=self.config['simulation_steps']-15)
                response_summary['shock_impact'] = shock_impact
                response_summary['policy_effect'] = policy_effect
                
                experiment_results[f'response_{scenario}_{policy}'] = response_summary
        
        # Generate visualizations
        print("Generating Digital Twin visualizations...")
        vis_paths = self.digital_twin.visualize_simulation(
            os.path.join(self.config['visualizations_dir'], 'digital_twin')
        )
        experiment_results['visualizations'] = vis_paths
        
        # Calculate resilience metrics
        resilience_metrics = self.digital_twin.calculate_resilience_metrics()
        experiment_results['resilience_metrics'] = resilience_metrics
        
        # Export simulation data
        output_file = os.path.join(self.config['results_dir'], 'digital_twin_simulation.csv')
        self.digital_twin.export_simulation_data(output_file)
        
        return experiment_results
    
    def run_resilience_scoring_experiments(self, data):
        """
        Run experiments using the Resilience Scoring Engine.
        
        Args:
            data (dict): Economic data for analysis
            
        Returns:
            dict: Experiment results
        """
        print("Running Resilience Scoring Engine experiments...")
        
        # Initialize resilience scoring engine
        self.resilience_engine = ResilienceScoringEngine()
        
        # Prepare data for resilience scoring
        if 'synthetic_data' in data:
            # Use synthetic data directly
            df = generate_synthetic_data(n_samples=100, n_features=12)
        else:
            # Convert real data to DataFrame
            records = []
            for entity, values in data['real_data'].items():
                record = {'entity': entity}
                record.update(values)
                records.append(record)
            df = pd.DataFrame(records)
        
        # Ensure resilience_score column exists
        if 'resilience_score' not in df.columns:
            # Generate synthetic resilience scores if needed
            print("Generating synthetic resilience scores...")
            features = df.drop(columns=['entity']).columns
            weights = {feature: np.random.uniform(-1, 1) for feature in features}
            
            # Calculate weighted sum
            df['resilience_score'] = 0
            for feature, weight in weights.items():
                df['resilience_score'] += df[feature] * weight
            
            # Normalize to 0-100 scale
            min_score = df['resilience_score'].min()
            max_score = df['resilience_score'].max()
            df['resilience_score'] = (df['resilience_score'] - min_score) / (max_score - min_score) * 100
        
        # Split data for training and testing
        train_df = df.sample(frac=0.8, random_state=self.config['random_seed'])
        test_df = df.drop(train_df.index)
        
        # Train resilience model
        print("Training resilience scoring model...")
        performance = self.resilience_engine.train(train_df.drop(columns=['entity']))
        
        # Identify key factors
        key_factors = self.resilience_engine.identify_key_factors(
            test_df.drop(columns=['entity', 'resilience_score'])
        )
        
        # Generate explanations
        vis_dir = os.path.join(self.config['visualizations_dir'], 'resilience')
        explanations = self.resilience_engine.explain_predictions(
            test_df.drop(columns=['entity', 'resilience_score']), 
            vis_dir
        )
        
        # Visualize model performance
        vis_paths = self.resilience_engine.visualize_model_performance(
            test_df.drop(columns=['entity']), 
            output_dir=vis_dir
        )
        
        # Save model
        model_path = os.path.join(self.config['results_dir'], 'resilience_model.joblib')
        self.resilience_engine.save_model(model_path)
        
        # Analyze policy impact
        # Create synthetic before/after policy data
        base_data = test_df.drop(columns=['entity', 'resilience_score']).copy()
        policy_data = base_data.copy()
        
        # Simulate policy effect (increase in positive factors, decrease in negative factors)
        for col in policy_data.columns:
            if col in key_factors['top_factors']:
                importance = key_factors['top_factors'][col]
                if importance > 0:
                    # Positive factor - increase by 10-20%
                    policy_data[col] *= (1 + np.random.uniform(0.1, 0.2))
                else:
                    # Negative factor - decrease by 10-20%
                    policy_data[col] *= (1 - np.random.uniform(0.1, 0.2))
        
        # Analyze impact
        policy_impact = self.resilience_engine.analyze_policy_impact(
            base_data, policy_data, 'fiscal_stimulus'
        )
        
        return {
            'model_performance': performance,
            'key_factors': key_factors,
            'explanations': explanations,
            'visualization_paths': vis_paths,
            'policy_impact': policy_impact
        }
    
    def run_marl_experiments(self):
        """
        Run experiments using the Multi-Agent Reinforcement Learning framework.
        
        Returns:
            dict: Experiment results
        """
        print("Running MARL framework experiments...")
        
        # Initialize MARL environment
        self.marl_environment = marl_env()
        
        # Run multiple trials
        trial_results = []
        
        for trial in range(self.config['num_trials']):
            print(f"Running MARL trial {trial+1}/{self.config['num_trials']}...")
            
            # Reset environment
            observations, _ = self.marl_environment.reset()
            
            # Track metrics
            episode_rewards = {agent: 0 for agent in self.marl_environment.agents}
            step_history = []
            
            # Run episode
            for step in range(self.config['simulation_steps']):
                # Simple action selection (random for now)
                actions = {
                    agent: self.marl_environment.action_spaces[agent].sample() 
                    for agent in self.marl_environment.agents
                }
                
                # Execute actions
                observations, rewards, terminations, truncations, infos = self.marl_environment.step(actions)
                
                # Update metrics
                for agent, reward in rewards.items():
                    episode_rewards[agent] += reward
                
                # Record step data
                step_data = {
                    'step': step,
                    'actions': {agent: action.tolist() for agent, action in actions.items()},
                    'rewards': rewards,
                    'observations': {agent: obs.tolist() for agent, obs in observations.items()}
                }
                step_history.append(step_data)
                
                # Check if episode is done
                if all(terminations.values()) or all(truncations.values()):
                    break
            
            # Record trial results
            trial_results.append({
                'trial': trial,
                'total_rewards': episode_rewards,
                'steps': len(step_history),
                'step_history': step_history
            })
        
        # Analyze results
        total_rewards_by_agent = {}
        for agent in self.marl_environment.agents:
            total_rewards_by_agent[agent] = [trial['total_rewards'][agent] for trial in trial_results]
        
        # Generate visualizations
        vis_dir = os.path.join(self.config['visualizations_dir'], 'marl')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Plot rewards by agent
        plt.figure(figsize=(12, 8))
        for agent, rewards in total_rewards_by_agent.items():
            plt.plot(range(1, self.config['num_trials']+1), rewards, marker='o', label=agent)
        plt.title('Total Rewards by Agent Across Trials')
        plt.xlabel('Trial')
        plt.ylabel('Total Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        rewards_path = os.path.join(vis_dir, 'agent_rewards.png')
        plt.savefig(rewards_path)
        plt.close()
        
        # Save results to file
        results_path = os.path.join(self.config['results_dir'], 'marl_results.json')
        with open(results_path, 'w') as f:
            # Convert all numpy types to Python native types before serialization
            serializable_data = self._make_serializable({
                'trials': trial_results,
                'summary': {
                    'mean_rewards': {agent: float(np.mean(rewards)) for agent, rewards in total_rewards_by_agent.items()},
                    'std_rewards': {agent: float(np.std(rewards)) for agent, rewards in total_rewards_by_agent.items()}
                }
            })
            json.dump(serializable_data, f, indent=2)
        
        return {
            'trials': trial_results,
            'rewards_by_agent': total_rewards_by_agent,
            'visualization_paths': [rewards_path]
        }
    
    def run_all_experiments(self):
        """
        Run all experiments and collect results.
        
        Returns:
            dict: All experiment results
        """
        print("Starting ResiliAI framework experiments...")
        
        # Prepare data
        data = self.prepare_data()
        
        # Run Digital Twin experiments
        if 'synthetic_data' in data:
            digital_twin_results = self.run_digital_twin_experiments(data['synthetic_data'])
        else:
            digital_twin_results = self.run_digital_twin_experiments(data['real_data'])
        
        # Run Resilience Scoring experiments
        resilience_results = self.run_resilience_scoring_experiments(data)
        
        # Run MARL experiments
        marl_results = self.run_marl_experiments()
        
        # Combine all results
        self.results = {
            'data': data,
            'digital_twin': digital_twin_results,
            'resilience_scoring': resilience_results,
            'marl': marl_results,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save combined results
        results_path = os.path.join(self.config['results_dir'], 'all_results.json')
        with open(results_path, 'w') as f:
            # Convert non-serializable objects to strings
            serializable_results = self._make_serializable(self.results)
            json.dump(serializable_results, f, indent=2)
        
        print(f"All experiments completed. Results saved to {results_path}")
        
        return self.results
    
    def _make_serializable(self, obj):
        """
        Convert non-serializable objects to serializable format.
        
        Args:
            obj: Object to convert
            
        Returns:
            Serializable version of the object
        """
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.ndarray, np.number)):
            return obj.tolist()
        elif isinstance(obj, (pd.DataFrame, pd.Series)):
            return obj.to_dict()
        elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
            return int(obj)
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj
    
    def generate_summary_report(self):
        """
        Generate a summary report of all experiment results.
        
        Returns:
            str: Path to the generated report
        """
        if not self.results:
            print("No results available. Run experiments first.")
            return None
        
        print("Generating summary report...")
        
        # Create report directory
        report_dir = os.path.join(self.config['results_dir'], 'report')
        os.makedirs(report_dir, exist_ok=True)
        
        # Generate report content
        report_content = []
        report_content.append("# ResiliAI Framework Experiment Results")
        report_content.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Digital Twin results
        report_content.append("## Digital Twin Economy Simulation Results")
        
        if 'digital_twin' in self.results:
            dt_results = self.results['digital_twin']
            
            # Resilience metrics
            if 'resilience_metrics' in dt_results:
                report_content.append("### Resilience Metrics by Entity")
                metrics_table = []
                metrics_table.append("| Entity | GDP Recovery Rate | Unemployment Volatility | Resilience Score | Shock Count |")
                metrics_table.append("| ------ | ----------------- | ---------------------- | ---------------- | ----------- |")
                
                for entity, metrics in dt_results['resilience_metrics'].items():
                    metrics_table.append(
                        f"| {entity} | {metrics.get('gdp_recovery_rate', 'N/A'):.2f} | "
                        f"{metrics.get('unemployment_volatility', 'N/A'):.2f} | "
                        f"{metrics.get('resilience_score', 'N/A'):.2f} | "
                        f"{metrics.get('shock_count', 'N/A')} |"
                    )
                
                report_content.extend(metrics_table)
                report_content.append("")
            
            # Visualizations
            if 'visualizations' in dt_results:
                report_content.append("### Digital Twin Visualizations")
                report_content.append("The following visualizations were generated:")
                
                for path in dt_results['visualizations']:
                    filename = os.path.basename(path)
                    report_content.append(f"- {filename}")
                
                report_content.append("")
        
        # Resilience Scoring results
        report_content.append("## Resilience Scoring Engine Results")
        
        if 'resilience_scoring' in self.results:
            rs_results = self.results['resilience_scoring']
            
            # Model performance
            if 'model_performance' in rs_results:
                perf = rs_results['model_performance']
                report_content.append("### Model Performance")
                report_content.append(f"- R² Score: {perf.get('r2', 'N/A'):.3f}")
                report_content.append(f"- RMSE: {perf.get('rmse', 'N/A'):.3f}")
                report_content.append("")
            
            # Key factors
            if 'key_factors' in rs_results:
                report_content.append("### Key Factors Affecting Resilience")
                factors_table = []
                factors_table.append("| Factor | Importance |")
                factors_table.append("| ------ | ---------- |")
                
                for factor, importance in rs_results['key_factors'].get('factor_ranking', {}).items():
                    factors_table.append(f"| {factor} | {importance:.4f} |")
                
                report_content.extend(factors_table)
                report_content.append("")
            
            # Policy impact
            if 'policy_impact' in rs_results:
                impact = rs_results['policy_impact']
                report_content.append(f"### Policy Impact Analysis: {impact.get('policy_name', 'Unknown Policy')}")
                report_content.append(f"- Mean Resilience Change: {impact.get('mean_resilience_change', 'N/A'):.2f}")
                report_content.append(f"- Positive Impact Percentage: {impact.get('positive_impact_percentage', 'N/A'):.1f}%")
                
                report_content.append("\n#### Most Influential Changes:")
                for factor, correlation in impact.get('most_influential_changes', []):
                    report_content.append(f"- {factor}: correlation = {correlation:.3f}")
                
                report_content.append("")
        
        # MARL results
        report_content.append("## Multi-Agent Reinforcement Learning Results")
        
        if 'marl' in self.results:
            marl_results = self.results['marl']
            
            # Rewards by agent
            if 'rewards_by_agent' in marl_results:
                report_content.append("### Agent Performance")
                rewards_table = []
                rewards_table.append("| Agent | Mean Reward | Std Deviation |")
                rewards_table.append("| ----- | ----------- | ------------- |")
                
                for agent, rewards in marl_results['rewards_by_agent'].items():
                    rewards_table.append(
                        f"| {agent} | {np.mean(rewards):.2f} | {np.std(rewards):.2f} |"
                    )
                
                report_content.extend(rewards_table)
                report_content.append("")
        
        # Write report to file
        report_path = os.path.join(report_dir, 'experiment_summary.md')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_content))
        
        print(f"Summary report generated at {report_path}")
        
        return report_path


def main():
    """
    Main function to run all experiments.
    """
    # Configure experiment runner
    config = {
        'data_dir': '/home/ubuntu/ESI2025_ResiliAI/data',
        'results_dir': '/home/ubuntu/ESI2025_ResiliAI/results',
        'visualizations_dir': '/home/ubuntu/ESI2025_ResiliAI/results/visualizations',
        'random_seed': 42,
        'experiment_scenarios': ['pandemic', 'financial_crisis', 'climate_shock'],
        'policy_interventions': ['fiscal_stimulus', 'monetary_easing', 'targeted_support'],
        'simulation_steps': 60,
        'num_trials': 5
    }
    
    # Create directories
    os.makedirs(config['data_dir'], exist_ok=True)
    os.makedirs(config['results_dir'], exist_ok=True)
    os.makedirs(config['visualizations_dir'], exist_ok=True)
    
    # Initialize and run experiments
    runner = ExperimentRunner(config)
    results = runner.run_all_experiments()
    
    # Generate summary report
    report_path = runner.generate_summary_report()
    
    print(f"Experiments completed. Summary report available at {report_path}")


if __name__ == "__main__":
    main()
