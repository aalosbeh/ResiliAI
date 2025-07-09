"""
Digital Twin Economy Simulation for ResiliAI Framework

This module implements a digital twin of an economy, simulating real-world economic
conditions and allowing for policy experimentation without affecting real markets.

Author: Anas ALsobeh, Raneem Alkurdi
Date: July 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import networkx as nx
from datetime import datetime, timedelta


class DigitalTwinEconomy:
    """
    A digital twin simulation of an economy that replicates real-world economic conditions
    and allows for policy experimentation.
    """
    
    def __init__(self, initial_data, config=None):
        """
        Initialize the digital twin with economic data.
        
        Args:
            initial_data (dict): Initial economic data by country/region
            config (dict, optional): Configuration parameters for the simulation
        """
        self.data = initial_data
        default_config = {
            'simulation_steps': 60,  # Monthly steps for 5 years
            'shock_probability': 0.1,  # Probability of economic shock per step
            'policy_effectiveness': 0.7,  # Effectiveness of policy interventions
            'recovery_rate': 0.05,  # Natural recovery rate per step
            'contagion_factor': 0.3,  # Economic contagion between connected entities
            'random_seed': 42  # For reproducibility
        }
        
        # Merge provided config with defaults
        self.config = default_config.copy()
        if config:
            self.config.update(config)
        
        # Set random seed for reproducibility
        np.random.seed(self.config['random_seed'])
        
        # Initialize economic network (countries/regions as nodes)
        self.network = self._initialize_network()
        
        # Initialize time series data storage
        self.time_series = {}
        self._initialize_time_series()
        
        # Track applied policies
        self.applied_policies = []
        
        # Current simulation step
        self.current_step = 0
        
    def _initialize_network(self):
        """
        Initialize the economic network with countries/regions as nodes.
        
        Returns:
            nx.Graph: Network graph of economic entities
        """
        G = nx.Graph()
        
        # Add nodes (countries/regions)
        for entity, data in self.data.items():
            G.add_node(entity, **data)
        
        # Add edges (economic connections)
        # In a real implementation, these would be based on trade data, financial flows, etc.
        entities = list(self.data.keys())
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities):
                if i < j:  # Avoid duplicate edges
                    # Randomly assign connection strength based on economic size
                    size1 = self.data[entity1].get('total_fiscal_response', 1)
                    size2 = self.data[entity2].get('total_fiscal_response', 1)
                    connection_strength = np.random.beta(2, 5) * np.sqrt(size1 * size2)
                    
                    if connection_strength > 0.2:  # Only add significant connections
                        G.add_edge(entity1, entity2, weight=connection_strength)
        
        return G
    
    def _initialize_time_series(self):
        """
        Initialize time series data storage for tracking economic indicators over time.
        """
        # Initialize with current values
        for entity, data in self.data.items():
            self.time_series[entity] = {
                'gdp': [100],  # Normalized to 100 at start
                'unemployment': [data.get('unemployment_rate', 5)],
                'inflation': [data.get('inflation_rate', 2)],
                'fiscal_response': [data.get('total_fiscal_response', 0)],
                'resilience_score': [data.get('resilience_score', 50)]
            }
    
    def apply_shock(self, shock_type='pandemic', severity=0.3, affected_entities=None):
        """
        Apply an economic shock to the simulation.
        
        Args:
            shock_type (str): Type of shock (pandemic, financial, climate, etc.)
            severity (float): Severity of the shock (0-1)
            affected_entities (list, optional): Specific entities to affect, or None for all
        
        Returns:
            dict: Impact of the shock on each entity
        """
        impact = {}
        
        # Default to all entities if none specified
        if affected_entities is None:
            affected_entities = list(self.data.keys())
        
        # Calculate base impact parameters based on shock type
        if shock_type == 'pandemic':
            gdp_impact = -severity * 15  # GDP reduction up to 15%
            unemployment_impact = severity * 10  # Unemployment increase up to 10 percentage points
            inflation_impact = severity * 5  # Inflation increase up to 5 percentage points
        elif shock_type == 'financial':
            gdp_impact = -severity * 10
            unemployment_impact = severity * 7
            inflation_impact = -severity * 2  # Potential deflation in financial crisis
        elif shock_type == 'climate':
            gdp_impact = -severity * 8
            unemployment_impact = severity * 5
            inflation_impact = severity * 8  # Higher inflation due to supply disruptions
        else:  # Generic shock
            gdp_impact = -severity * 10
            unemployment_impact = severity * 6
            inflation_impact = severity * 4
        
        # Apply shock to each affected entity
        for entity in affected_entities:
            if entity not in self.data:
                continue
                
            # Adjust impact based on entity's resilience
            resilience = self.data[entity].get('resilience_score', 50) / 100
            adjusted_gdp_impact = gdp_impact * (1 - resilience)
            adjusted_unemployment_impact = unemployment_impact * (1 - resilience)
            adjusted_inflation_impact = inflation_impact * (1 - resilience)
            
            # Update current values
            current_gdp = self.time_series[entity]['gdp'][-1]
            current_unemployment = self.time_series[entity]['unemployment'][-1]
            current_inflation = self.time_series[entity]['inflation'][-1]
            
            new_gdp = current_gdp * (1 + adjusted_gdp_impact / 100)
            new_unemployment = current_unemployment + adjusted_unemployment_impact
            new_inflation = current_inflation + adjusted_inflation_impact
            
            # Ensure values stay in reasonable ranges
            new_gdp = max(new_gdp, current_gdp * 0.5)  # Limit GDP drop to 50%
            new_unemployment = min(max(new_unemployment, 0), 30)  # 0-30% range
            new_inflation = min(max(new_inflation, -5), 30)  # -5% to 30% range
            
            # Record impact
            impact[entity] = {
                'gdp_change': (new_gdp - current_gdp) / current_gdp * 100,
                'unemployment_change': new_unemployment - current_unemployment,
                'inflation_change': new_inflation - current_inflation
            }
            
            # Update time series
            self.time_series[entity]['gdp'].append(new_gdp)
            self.time_series[entity]['unemployment'].append(new_unemployment)
            self.time_series[entity]['inflation'].append(new_inflation)
            self.time_series[entity]['resilience_score'].append(self.time_series[entity]['resilience_score'][-1])
            self.time_series[entity]['fiscal_response'].append(self.time_series[entity]['fiscal_response'][-1])
        
        # Propagate shock through network
        self._propagate_shock(impact, affected_entities)
        
        return impact
    
    def _propagate_shock(self, initial_impact, affected_entities, propagation_rounds=2):
        """
        Propagate economic shock through the network to connected entities.
        
        Args:
            initial_impact (dict): Initial impact of the shock
            affected_entities (list): Initially affected entities
            propagation_rounds (int): Number of propagation rounds
        """
        # Skip if no network connections
        if len(self.network.edges()) == 0:
            return
        
        # Entities already affected by the initial shock
        impacted = set(affected_entities)
        
        for _ in range(propagation_rounds):
            new_impacts = {}
            
            # For each already impacted entity
            for entity in impacted:
                # Get its neighbors
                for neighbor in self.network.neighbors(entity):
                    if neighbor in impacted:
                        continue
                    
                    # Calculate propagated impact based on connection strength
                    edge_weight = self.network[entity][neighbor]['weight']
                    contagion = self.config['contagion_factor'] * edge_weight
                    
                    # Propagate a fraction of the impact
                    if neighbor not in new_impacts:
                        new_impacts[neighbor] = {
                            'gdp_change': 0,
                            'unemployment_change': 0,
                            'inflation_change': 0
                        }
                    
                    new_impacts[neighbor]['gdp_change'] += initial_impact[entity]['gdp_change'] * contagion
                    new_impacts[neighbor]['unemployment_change'] += initial_impact[entity]['unemployment_change'] * contagion
                    new_impacts[neighbor]['inflation_change'] += initial_impact[entity]['inflation_change'] * contagion
            
            # Apply the propagated impacts
            for entity, impact in new_impacts.items():
                current_gdp = self.time_series[entity]['gdp'][-1]
                current_unemployment = self.time_series[entity]['unemployment'][-1]
                current_inflation = self.time_series[entity]['inflation'][-1]
                
                new_gdp = current_gdp * (1 + impact['gdp_change'] / 100)
                new_unemployment = current_unemployment + impact['unemployment_change']
                new_inflation = current_inflation + impact['inflation_change']
                
                # Ensure values stay in reasonable ranges
                new_gdp = max(new_gdp, current_gdp * 0.8)  # Limit secondary GDP drop
                new_unemployment = min(max(new_unemployment, 0), 25)
                new_inflation = min(max(new_inflation, -3), 20)
                
                # Update time series
                self.time_series[entity]['gdp'][-1] = new_gdp
                self.time_series[entity]['unemployment'][-1] = new_unemployment
                self.time_series[entity]['inflation'][-1] = new_inflation
                
                # Add to impacted set for next round
                impacted.add(entity)
    
    def apply_policy(self, policy_type, target_entities=None, intensity=0.5):
        """
        Apply a policy intervention to the simulation.
        
        Args:
            policy_type (str): Type of policy (fiscal_stimulus, monetary_easing, etc.)
            target_entities (list, optional): Specific entities to target, or None for all
            intensity (float): Intensity of the policy (0-1)
            
        Returns:
            dict: Effect of the policy on each entity
        """
        effect = {}
        
        # Default to all entities if none specified
        if target_entities is None:
            target_entities = list(self.data.keys())
        
        # Calculate base effect parameters based on policy type
        if policy_type == 'fiscal_stimulus':
            gdp_effect = intensity * 5  # GDP increase up to 5%
            unemployment_effect = -intensity * 3  # Unemployment decrease up to 3 percentage points
            inflation_effect = intensity * 2  # Inflation increase up to 2 percentage points
            fiscal_response_increase = intensity * 10  # Increase in fiscal response measure
        elif policy_type == 'monetary_easing':
            gdp_effect = intensity * 3
            unemployment_effect = -intensity * 2
            inflation_effect = intensity * 3
            fiscal_response_increase = 0  # No direct change in fiscal response
        elif policy_type == 'targeted_support':
            gdp_effect = intensity * 2
            unemployment_effect = -intensity * 4
            inflation_effect = intensity * 1
            fiscal_response_increase = intensity * 5
        else:  # Generic policy
            gdp_effect = intensity * 3
            unemployment_effect = -intensity * 2
            inflation_effect = intensity * 1.5
            fiscal_response_increase = intensity * 5
        
        # Apply policy to each target entity
        for entity in target_entities:
            if entity not in self.data:
                continue
                
            # Adjust effect based on policy effectiveness
            effectiveness = self.config['policy_effectiveness']
            adjusted_gdp_effect = gdp_effect * effectiveness
            adjusted_unemployment_effect = unemployment_effect * effectiveness
            adjusted_inflation_effect = inflation_effect * effectiveness
            
            # Update current values
            current_gdp = self.time_series[entity]['gdp'][-1]
            current_unemployment = self.time_series[entity]['unemployment'][-1]
            current_inflation = self.time_series[entity]['inflation'][-1]
            current_fiscal_response = self.time_series[entity]['fiscal_response'][-1]
            
            new_gdp = current_gdp * (1 + adjusted_gdp_effect / 100)
            new_unemployment = current_unemployment + adjusted_unemployment_effect
            new_inflation = current_inflation + adjusted_inflation_effect
            new_fiscal_response = current_fiscal_response + fiscal_response_increase
            
            # Ensure values stay in reasonable ranges
            new_unemployment = min(max(new_unemployment, 0), 30)
            new_inflation = min(max(new_inflation, -5), 30)
            
            # Record effect
            effect[entity] = {
                'gdp_change': (new_gdp - current_gdp) / current_gdp * 100,
                'unemployment_change': new_unemployment - current_unemployment,
                'inflation_change': new_inflation - current_inflation,
                'fiscal_response_change': new_fiscal_response - current_fiscal_response
            }
            
            # Update time series
            self.time_series[entity]['gdp'][-1] = new_gdp
            self.time_series[entity]['unemployment'][-1] = new_unemployment
            self.time_series[entity]['inflation'][-1] = new_inflation
            self.time_series[entity]['fiscal_response'][-1] = new_fiscal_response
            
            # Update resilience score based on policy
            current_resilience = self.time_series[entity]['resilience_score'][-1]
            resilience_boost = intensity * effectiveness * 5  # Up to 5 point boost
            new_resilience = min(current_resilience + resilience_boost, 100)
            self.time_series[entity]['resilience_score'][-1] = new_resilience
        
        # Record the applied policy
        self.applied_policies.append({
            'step': self.current_step,
            'type': policy_type,
            'targets': target_entities,
            'intensity': intensity,
            'effect': effect
        })
        
        return effect
    
    def simulate_step(self, apply_random_shocks=True):
        """
        Simulate one time step in the digital twin economy.
        
        Args:
            apply_random_shocks (bool): Whether to apply random shocks
            
        Returns:
            dict: Summary of changes in this step
        """
        self.current_step += 1
        summary = {'step': self.current_step, 'shocks': [], 'natural_recovery': {}}
        
        # Apply random shocks if enabled
        if apply_random_shocks and np.random.random() < self.config['shock_probability']:
            shock_type = np.random.choice(['pandemic', 'financial', 'climate'])
            severity = np.random.beta(2, 5)  # Beta distribution for severity
            
            # For targeted shocks, select a random subset of entities
            if np.random.random() < 0.3:  # 30% chance of targeted shock
                num_targets = max(1, int(len(self.data) * np.random.beta(2, 5)))
                targets = np.random.choice(list(self.data.keys()), size=num_targets, replace=False)
            else:
                targets = None
                
            impact = self.apply_shock(shock_type, severity, targets)
            summary['shocks'].append({
                'type': shock_type,
                'severity': severity,
                'targets': targets,
                'impact': impact
            })
        
        # Natural recovery and fluctuations for all entities
        for entity in self.data.keys():
            # Get current values
            current_gdp = self.time_series[entity]['gdp'][-1]
            current_unemployment = self.time_series[entity]['unemployment'][-1]
            current_inflation = self.time_series[entity]['inflation'][-1]
            
            # Calculate recovery and random fluctuations
            recovery_rate = self.config['recovery_rate']
            gdp_recovery = (100 - current_gdp) * recovery_rate  # Recovery toward baseline of 100
            unemployment_recovery = -current_unemployment * recovery_rate * 0.1  # Slow recovery toward 0
            inflation_recovery = (2 - current_inflation) * recovery_rate * 0.2  # Recovery toward 2%
            
            # Add random fluctuations
            gdp_fluctuation = np.random.normal(0, 1)  # Normal distribution with mean 0, std 1
            unemployment_fluctuation = np.random.normal(0, 0.2)
            inflation_fluctuation = np.random.normal(0, 0.3)
            
            # Calculate new values
            new_gdp = current_gdp * (1 + (gdp_recovery + gdp_fluctuation) / 100)
            new_unemployment = current_unemployment + unemployment_recovery + unemployment_fluctuation
            new_inflation = current_inflation + inflation_recovery + inflation_fluctuation
            
            # Ensure values stay in reasonable ranges
            new_unemployment = min(max(new_unemployment, 0), 30)
            new_inflation = min(max(new_inflation, -5), 30)
            
            # Record changes
            summary['natural_recovery'][entity] = {
                'gdp_change': (new_gdp - current_gdp) / current_gdp * 100,
                'unemployment_change': new_unemployment - current_unemployment,
                'inflation_change': new_inflation - current_inflation
            }
            
            # Add new values to time series
            self.time_series[entity]['gdp'].append(new_gdp)
            self.time_series[entity]['unemployment'].append(new_unemployment)
            self.time_series[entity]['inflation'].append(new_inflation)
            
            # Carry forward other metrics
            self.time_series[entity]['resilience_score'].append(self.time_series[entity]['resilience_score'][-1])
            self.time_series[entity]['fiscal_response'].append(self.time_series[entity]['fiscal_response'][-1])
        
        return summary
    
    def run_simulation(self, steps=None, policies=None):
        """
        Run the simulation for multiple steps with optional policy interventions.
        
        Args:
            steps (int, optional): Number of steps to simulate, defaults to config value
            policies (list, optional): List of policy interventions to apply
            
        Returns:
            dict: Summary of the simulation run
        """
        if steps is None:
            steps = self.config['simulation_steps']
            
        simulation_summary = {
            'start_step': self.current_step,
            'end_step': self.current_step + steps,
            'shocks': [],
            'policies': [],
            'final_state': {}
        }
        
        # Schedule policies if provided
        policy_schedule = {}
        if policies:
            for policy in policies:
                step = policy.get('step', 0)
                if step in policy_schedule:
                    policy_schedule[step].append(policy)
                else:
                    policy_schedule[step] = [policy]
        
        # Run simulation steps
        for i in range(steps):
            step = self.current_step + 1  # Next step to simulate
            
            # Apply scheduled policies for this step
            if step in policy_schedule:
                for policy in policy_schedule[step]:
                    effect = self.apply_policy(
                        policy['type'], 
                        policy.get('targets'), 
                        policy.get('intensity', 0.5)
                    )
                    simulation_summary['policies'].append({
                        'step': step,
                        'type': policy['type'],
                        'targets': policy.get('targets'),
                        'intensity': policy.get('intensity', 0.5),
                        'effect': effect
                    })
            
            # Simulate the step
            step_summary = self.simulate_step()
            
            # Record any shocks
            if step_summary['shocks']:
                simulation_summary['shocks'].extend(step_summary['shocks'])
        
        # Record final state
        for entity in self.data.keys():
            simulation_summary['final_state'][entity] = {
                'gdp': self.time_series[entity]['gdp'][-1],
                'unemployment': self.time_series[entity]['unemployment'][-1],
                'inflation': self.time_series[entity]['inflation'][-1],
                'resilience_score': self.time_series[entity]['resilience_score'][-1],
                'fiscal_response': self.time_series[entity]['fiscal_response'][-1]
            }
        
        return simulation_summary
    
    def calculate_resilience_metrics(self):
        """
        Calculate economic resilience metrics based on simulation results.
        
        Returns:
            dict: Resilience metrics by entity
        """
        metrics = {}
        
        for entity in self.data.keys():
            # Get time series data
            gdp = np.array(self.time_series[entity]['gdp'])
            unemployment = np.array(self.time_series[entity]['unemployment'])
            
            # Skip if not enough data points
            if len(gdp) < 3:
                continue
                
            # Calculate metrics
            
            # 1. GDP recovery rate after shocks
            gdp_diffs = np.diff(gdp)
            negative_shocks = np.where(gdp_diffs < -1)[0]  # Identify significant negative shocks
            
            recovery_rates = []
            for shock_idx in negative_shocks:
                shock_size = abs(gdp_diffs[shock_idx])
                
                # Look for recovery in the next 12 steps (or fewer if not enough data)
                recovery_window = min(12, len(gdp) - shock_idx - 2)
                if recovery_window <= 0:
                    continue
                    
                post_shock = gdp[shock_idx+1:shock_idx+1+recovery_window]
                pre_shock = gdp[shock_idx]
                
                # Calculate recovery percentage
                max_recovery = max(post_shock) - min(post_shock)
                recovery_pct = max_recovery / shock_size if shock_size > 0 else 0
                recovery_rates.append(recovery_pct)
            
            # 2. Unemployment stability
            unemployment_volatility = np.std(unemployment)
            
            # 3. Overall resilience score
            avg_recovery = np.mean(recovery_rates) if recovery_rates else 0
            resilience_score = 50 + 25 * avg_recovery - 5 * unemployment_volatility
            resilience_score = min(max(resilience_score, 0), 100)  # Ensure 0-100 range
            
            metrics[entity] = {
                'gdp_recovery_rate': avg_recovery,
                'unemployment_volatility': unemployment_volatility,
                'resilience_score': resilience_score,
                'shock_count': len(negative_shocks)
            }
        
        return metrics
    
    def visualize_simulation(self, output_dir, entities=None, metrics=None):
        """
        Create visualizations of simulation results.
        
        Args:
            output_dir (str): Directory to save visualizations
            entities (list, optional): Specific entities to visualize, or None for all
            metrics (list, optional): Specific metrics to visualize, or None for all
            
        Returns:
            list: Paths to saved visualization files
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        if entities is None:
            entities = list(self.data.keys())
            
            # If too many entities, select a representative sample
            if len(entities) > 10:
                entities = np.random.choice(entities, 10, replace=False)
        
        if metrics is None:
            metrics = ['gdp', 'unemployment', 'inflation', 'resilience_score']
        
        visualization_paths = []
        
        # 1. Time series plots for each metric
        for metric in metrics:
            plt.figure(figsize=(12, 8))
            
            for entity in entities:
                if entity in self.time_series and metric in self.time_series[entity]:
                    plt.plot(self.time_series[entity][metric], label=entity)
            
            plt.title(f'{metric.replace("_", " ").title()} Over Time')
            plt.xlabel('Simulation Step')
            plt.ylabel(metric.replace('_', ' ').title())
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Mark policy interventions
            for policy in self.applied_policies:
                if any(entity in policy['targets'] for entity in entities) or policy['targets'] is None:
                    plt.axvline(x=policy['step'], color='g', linestyle='--', alpha=0.5)
                    plt.text(policy['step'], plt.ylim()[1]*0.9, policy['type'], rotation=90, alpha=0.7)
            
            # Save the figure
            fig_path = os.path.join(output_dir, f'simulation_{metric}.png')
            plt.savefig(fig_path)
            visualization_paths.append(fig_path)
            plt.close()
        
        # 2. Economic network visualization
        if len(self.network.nodes()) > 0:
            plt.figure(figsize=(12, 10))
            
            # Get resilience scores for node colors
            resilience_scores = {}
            for entity in self.network.nodes():
                if entity in self.time_series and 'resilience_score' in self.time_series[entity]:
                    resilience_scores[entity] = self.time_series[entity]['resilience_score'][-1]
                else:
                    resilience_scores[entity] = 50  # Default value
            
            # Get node positions using spring layout
            pos = nx.spring_layout(self.network, seed=42)
            
            # Draw nodes with colors based on resilience
            node_colors = [resilience_scores[node] for node in self.network.nodes()]
            nodes = nx.draw_networkx_nodes(
                self.network, pos, 
                node_color=node_colors, 
                cmap=plt.cm.viridis,
                node_size=500,
                alpha=0.8
            )
            
            # Draw edges with width based on weight
            edge_weights = [self.network[u][v]['weight'] * 3 for u, v in self.network.edges()]
            edges = nx.draw_networkx_edges(
                self.network, pos,
                width=edge_weights,
                alpha=0.5,
                edge_color='gray'
            )
            
            # Draw labels
            nx.draw_networkx_labels(self.network, pos, font_size=10)
            
            # Add colorbar for resilience scores
            plt.colorbar(nodes, label='Resilience Score')
            
            plt.title('Economic Network with Resilience Scores')
            plt.axis('off')
            
            # Save the figure
            fig_path = os.path.join(output_dir, 'economic_network.png')
            plt.savefig(fig_path)
            visualization_paths.append(fig_path)
            plt.close()
        
        # 3. Resilience metrics comparison
        resilience_metrics = self.calculate_resilience_metrics()
        if resilience_metrics and len(entities) > 1:
            metrics_to_plot = ['gdp_recovery_rate', 'unemployment_volatility', 'resilience_score']
            
            for metric in metrics_to_plot:
                plt.figure(figsize=(10, 6))
                
                values = [resilience_metrics[entity][metric] for entity in entities 
                         if entity in resilience_metrics]
                entity_labels = [entity for entity in entities 
                               if entity in resilience_metrics]
                
                if not values:
                    plt.close()
                    continue
                
                plt.bar(entity_labels, values)
                plt.title(f'{metric.replace("_", " ").title()} by Entity')
                plt.ylabel(metric.replace('_', ' ').title())
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                # Save the figure
                fig_path = os.path.join(output_dir, f'resilience_{metric}.png')
                plt.savefig(fig_path)
                visualization_paths.append(fig_path)
                plt.close()
        
        return visualization_paths
    
    def export_simulation_data(self, output_file):
        """
        Export simulation data to CSV for further analysis.
        
        Args:
            output_file (str): Path to save the CSV file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Prepare data for export
            export_data = []
            
            for entity in self.data.keys():
                for step in range(len(self.time_series[entity]['gdp'])):
                    row = {
                        'entity': entity,
                        'step': step,
                        'gdp': self.time_series[entity]['gdp'][step],
                        'unemployment': self.time_series[entity]['unemployment'][step],
                        'inflation': self.time_series[entity]['inflation'][step],
                        'resilience_score': self.time_series[entity]['resilience_score'][step],
                        'fiscal_response': self.time_series[entity]['fiscal_response'][step]
                    }
                    export_data.append(row)
            
            # Convert to DataFrame and save
            df = pd.DataFrame(export_data)
            df.to_csv(output_file, index=False)
            
            print(f"Successfully exported simulation data to {output_file}")
            return True
        except Exception as e:
            print(f"Error exporting simulation data: {str(e)}")
            return False


def main():
    """
    Main function to demonstrate the digital twin economy simulation.
    """
    # Sample initial data
    initial_data = {
        'USA': {
            'total_fiscal_response': 25.5,
            'above_the_line': 18.7,
            'below_the_line': 6.8,
            'unemployment_rate': 5.2,
            'inflation_rate': 2.1,
            'resilience_score': 75
        },
        'EU': {
            'total_fiscal_response': 20.1,
            'above_the_line': 14.3,
            'below_the_line': 5.8,
            'unemployment_rate': 7.5,
            'inflation_rate': 1.8,
            'resilience_score': 70
        },
        'China': {
            'total_fiscal_response': 15.8,
            'above_the_line': 10.2,
            'below_the_line': 5.6,
            'unemployment_rate': 4.0,
            'inflation_rate': 2.5,
            'resilience_score': 65
        },
        'Japan': {
            'total_fiscal_response': 22.0,
            'above_the_line': 16.5,
            'below_the_line': 5.5,
            'unemployment_rate': 3.0,
            'inflation_rate': 0.5,
            'resilience_score': 72
        },
        'UK': {
            'total_fiscal_response': 18.9,
            'above_the_line': 13.2,
            'below_the_line': 5.7,
            'unemployment_rate': 4.5,
            'inflation_rate': 2.0,
            'resilience_score': 68
        }
    }
    
    # Initialize digital twin
    digital_twin = DigitalTwinEconomy(initial_data)
    
    # Define policy interventions
    policies = [
        {
            'step': 10,
            'type': 'fiscal_stimulus',
            'targets': ['USA', 'EU', 'UK'],
            'intensity': 0.7
        },
        {
            'step': 20,
            'type': 'monetary_easing',
            'targets': None,  # All entities
            'intensity': 0.5
        },
        {
            'step': 35,
            'type': 'targeted_support',
            'targets': ['Japan', 'China'],
            'intensity': 0.8
        }
    ]
    
    # Run simulation
    simulation_summary = digital_twin.run_simulation(steps=60, policies=policies)
    
    # Calculate resilience metrics
    resilience_metrics = digital_twin.calculate_resilience_metrics()
    
    # Create visualizations
    vis_dir = "/home/ubuntu/ESI2025_ResiliAI/data/visualizations"
    visualization_paths = digital_twin.visualize_simulation(vis_dir)
    
    # Export simulation data
    output_file = "/home/ubuntu/ESI2025_ResiliAI/data/simulation_results.csv"
    digital_twin.export_simulation_data(output_file)
    
    print(f"Simulation completed with {len(simulation_summary['shocks'])} shocks and {len(simulation_summary['policies'])} policy interventions")
    print(f"Created {len(visualization_paths)} visualizations in {vis_dir}")
    print(f"Exported simulation data to {output_file}")


if __name__ == "__main__":
    main()
