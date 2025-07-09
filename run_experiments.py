"""
Main script to run the ResiliAI framework experiments and generate results

This script executes the complete experimental workflow for the ResiliAI framework,
including data preprocessing, MARL simulation, digital twin economy simulation,
and resilience scoring with explainable AI.

Author: Anas ALsobeh, Raneem Alkurdi
Date: July 2025
"""

import os
import sys
import time
import argparse
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the experiment runner
from src.experiment_runner import ExperimentRunner


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run ResiliAI framework experiments')
    
    parser.add_argument('--data_dir', type=str, default='/home/ubuntu/ESI2025_ResiliAI/data',
                        help='Directory containing input data')
    parser.add_argument('--results_dir', type=str, default='/home/ubuntu/ESI2025_ResiliAI/results',
                        help='Directory to save results')
    parser.add_argument('--visualizations_dir', type=str, 
                        default='/home/ubuntu/ESI2025_ResiliAI/results/visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--simulation_steps', type=int, default=60,
                        help='Number of steps in each simulation')
    parser.add_argument('--num_trials', type=int, default=5,
                        help='Number of trials for MARL experiments')
    parser.add_argument('--scenarios', type=str, nargs='+', 
                        default=['pandemic', 'financial_crisis', 'climate_shock'],
                        help='Shock scenarios to simulate')
    parser.add_argument('--policies', type=str, nargs='+',
                        default=['fiscal_stimulus', 'monetary_easing', 'targeted_support'],
                        help='Policy interventions to test')
    
    return parser.parse_args()


def main():
    """Main function to run experiments."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Print experiment configuration
    print("=" * 80)
    print("ResiliAI Framework Experiments")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data directory: {args.data_dir}")
    print(f"Results directory: {args.results_dir}")
    print(f"Visualizations directory: {args.visualizations_dir}")
    print(f"Random seed: {args.random_seed}")
    print(f"Simulation steps: {args.simulation_steps}")
    print(f"Number of trials: {args.num_trials}")
    print(f"Shock scenarios: {args.scenarios}")
    print(f"Policy interventions: {args.policies}")
    print("=" * 80)
    
    # Create configuration dictionary
    config = {
        'data_dir': args.data_dir,
        'results_dir': args.results_dir,
        'visualizations_dir': args.visualizations_dir,
        'random_seed': args.random_seed,
        'experiment_scenarios': args.scenarios,
        'policy_interventions': args.policies,
        'simulation_steps': args.simulation_steps,
        'num_trials': args.num_trials
    }
    
    # Create directories if they don't exist
    os.makedirs(config['data_dir'], exist_ok=True)
    os.makedirs(config['results_dir'], exist_ok=True)
    os.makedirs(config['visualizations_dir'], exist_ok=True)
    
    # Record start time
    start_time = time.time()
    
    # Initialize and run experiments
    runner = ExperimentRunner(config)
    
    try:
        # Run all experiments
        results = runner.run_all_experiments()
        
        # Generate summary report
        report_path = runner.generate_summary_report()
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print("=" * 80)
        print(f"Experiments completed successfully in {int(hours)}h {int(minutes)}m {int(seconds)}s")
        print(f"Summary report available at: {report_path}")
        print("=" * 80)
        
    except Exception as e:
        print("=" * 80)
        print(f"Error running experiments: {str(e)}")
        print("=" * 80)
        raise
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
