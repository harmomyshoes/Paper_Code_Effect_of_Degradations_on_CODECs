"""
Enhanced Training System with Per-Project Score Tracking

This module extends the RL training to track individual project scores
alongside the aggregated reward during training.
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime
from collections import defaultdict
import tempfile


class ProjectScoreTracker:
    """
    Tracks individual project scores during RL training.
    Stores scores as a side effect when reward function is called.
    """
    
    def __init__(self, project_names):
        self.project_names = project_names
        # Store scores for each training step
        self.project_scores_history = []
        # Store current scores (updated by reward function)
        self.current_scores = {}
        self.current_aggregated_score = None
        self.step_counter = 0
        
    def update_scores(self, scores_dict, aggregated_score):
        """Called by reward function to update current scores"""
        self.current_scores = scores_dict.copy()
        self.current_aggregated_score = aggregated_score
        
    def log_step(self, update_num, episode_num, timestep):
        """Log current scores for this step"""
        record = {
            'update': update_num,
            'episode': episode_num,
            'timestep': timestep,
            'aggregated_reward': self.current_aggregated_score,
        }
        # Add individual project scores
        for project_name in self.project_names:
            record[f'reward_{project_name}'] = self.current_scores.get(project_name, None)
        
        self.project_scores_history.append(record)
        self.step_counter += 1
        
    def get_current_project_scores(self):
        """Get current project scores as a formatted string"""
        if not self.current_scores:
            return "No scores available"
        
        scores_str = " | ".join([
            f"{name}: {score:.4f}" 
            for name, score in self.current_scores.items()
        ])
        return scores_str
    
    def save_history(self, save_dir, suffix=''):
        """Save project score history to CSV"""
        if not self.project_scores_history:
            print("No project score history to save")
            return None
            
        df = pd.DataFrame(self.project_scores_history)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if suffix:
            filename = f'project_scores_{timestamp}_{suffix}.csv'
        else:
            filename = f'project_scores_{timestamp}.csv'
        
        filepath = os.path.join(save_dir, filename)
        df.to_csv(filepath, index=False)
        
        print(f"\n{'='*80}")
        print(f"SAVED PROJECT SCORES!")
        print(f"{'='*80}")
        print(f"File: {filepath}")
        print(f"Total records: {len(df)}")
        print(f"Projects tracked: {', '.join(self.project_names)}")
        print(f"{'='*80}\n")
        
        return filepath
    
    def get_best_per_project(self):
        """Get best scores achieved per project"""
        if not self.project_scores_history:
            return {}
        
        df = pd.DataFrame(self.project_scores_history)
        best_scores = {}
        
        for project_name in self.project_names:
            col_name = f'reward_{project_name}'
            if col_name in df.columns:
                # Lower reward is better (1 - HAAQI)
                best_scores[project_name] = df[col_name].min()
        
        return best_scores


def create_enhanced_reward_function(project_manager, tracker, aggregation='mean'):
    """
    Factory function that creates a reward function with project tracking.
    
    Args:
        project_manager: ProjectManager instance
        tracker: ProjectScoreTracker instance
        aggregation: Aggregation method for scores
        
    Returns:
        Enhanced reward function that tracks project scores
    """
    
    def haaqi_reward_multi_fn_tracked(solution: np.ndarray, 
                                      is_normalised=True) -> float:
        """
        Enhanced reward function that tracks individual project scores.
        """
        # Import here to avoid circular imports
        from Optimiser.config import denormalize_action_FullTrack as denormalize_action
        import CODECbreakCode.Evaluator as Evaluator
        
        if is_normalised:
            solution = denormalize_action(solution)
        
        solution = list(solution)
        scores = {}
        
        # Evaluate each project
        for project_name, noise_generator in project_manager.projects.items():
            # Create unique temp file
            fd, degradated_filename = tempfile.mkstemp(
                prefix=f"dynC_{project_name}_", 
                suffix=".wav",
            )
            os.close(fd)
            
            try:
                # Generate degraded audio
                gener_audio = noise_generator.TestDynNoisedFullTrack(
                    solution,
                    degradated_filename,
                    isNormalised=False,
                    isCompensated=True,
                    foldpath=project_manager.base_dir
                )
                
                # Compress to MP3
                gener_audio_mp3 = Evaluator.Mp3LameLossyCompress(gener_audio, 64)
                
                # Measure HAAQI using the project-specific measurer
                haaqi_measurer = project_manager.haaqi_measurers[project_name]
                haaqi_score = haaqi_measurer.MeasureHAQQIOutput(gener_audio_mp3)
                scores[project_name] = 1 - haaqi_score
                
            except Exception as e:
                print(f"Error evaluating {project_name}: {e}")
                scores[project_name] = 0.0  # Worst score on error
                
            finally:
                # Clean up temp files
                for temp_file in [degradated_filename, gener_audio, gener_audio_mp3]:
                    try:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                    except OSError:
                        pass
        
        # Aggregate scores
        if not scores:
            final_score = 0.0
        else:
            score_values = list(scores.values())
            
            if aggregation == 'mean':
                final_score = np.mean(score_values)
            elif aggregation == 'sum':
                final_score = np.sum(score_values)
            elif aggregation == 'min':
                final_score = np.min(score_values)
            elif aggregation == 'max':
                final_score = np.max(score_values)
            else:
                final_score = np.mean(score_values)
        
        final_score = round(final_score, 3)
        
        # Update tracker with current scores
        tracker.update_scores(scores, final_score)
        
        # ✅ FIX: Actually log to history (this was missing!)
        record = {
            'aggregated_reward': final_score,
        }
        # Add individual project scores
        for project_name in tracker.project_names:
            record[f'reward_{project_name}'] = scores.get(project_name, None)
        
        tracker.project_scores_history.append(record)
        tracker.step_counter += 1
        
        return final_score
    
    return haaqi_reward_multi_fn_tracked


class EnhancedDataCollector:
    """
    Enhanced data collector that also tracks project scores.
    """
    
    def __init__(self, save_dir='rl_analysis_data', project_tracker=None):
        self.save_dir = save_dir
        self.actions = []
        self.rewards = []
        self.project_tracker = project_tracker
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print(f"Enhanced data collector initialized. Saving to: {save_dir}")
        
    def add(self, actions, rewards, update_num, episode_num):
        """Add batch of actions and rewards"""
        # Convert TF tensors to numpy
        if hasattr(actions, 'numpy'):
            actions = actions.numpy()
        if hasattr(rewards, 'numpy'):
            rewards = rewards.numpy()
            
        # Flatten if multi-dimensional
        if len(actions.shape) > 2:
            actions = actions.reshape(-1, actions.shape[-1])
        if len(rewards.shape) > 1:
            rewards = rewards.reshape(-1)
            
        self.actions.append(actions)
        self.rewards.append(rewards)
        
        # Log project scores for each timestep
        if self.project_tracker:
            num_steps = len(rewards)
            for t in range(num_steps):
                self.project_tracker.log_step(update_num, episode_num, t)
        
    def save(self, suffix=''):
        """Save collected data to files"""
        if len(self.actions) == 0:
            print("No data to save!")
            return None, None, None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save actions and rewards as before
        actions = np.vstack(self.actions)
        rewards = np.concatenate(self.rewards)
        
        if suffix:
            actions_file = os.path.join(self.save_dir, f'actions_{timestamp}_{suffix}.npy')
            rewards_file = os.path.join(self.save_dir, f'rewards_{timestamp}_{suffix}.npy')
        else:
            actions_file = os.path.join(self.save_dir, f'actions_{timestamp}.npy')
            rewards_file = os.path.join(self.save_dir, f'rewards_{timestamp}.npy')
        
        np.save(actions_file, actions)
        np.save(rewards_file, rewards)
        
        # Save project scores
        project_scores_file = None
        if self.project_tracker:
            project_scores_file = self.project_tracker.save_history(
                self.save_dir, 
                suffix=suffix
            )
        
        print(f"\n{'='*80}")
        print(f"SAVED ENHANCED DATA FOR ANALYSIS!")
        print(f"{'='*80}")
        print(f"Actions: {actions_file}")
        print(f"  Shape: {actions.shape}")
        print(f"Rewards: {rewards_file}")
        print(f"  Shape: {rewards.shape}")
        print(f"  Mean: {rewards.mean():.3f} ± {rewards.std():.3f}")
        if project_scores_file:
            print(f"Project Scores: {project_scores_file}")
        print(f"{'='*80}\n")
        
        return actions_file, rewards_file, project_scores_file


def analyze_project_performance(project_scores_file):
    """
    Analyze per-project performance from saved scores.
    
    Args:
        project_scores_file: Path to CSV with project scores
        
    Returns:
        Dictionary with analysis results
    """
    df = pd.DataFrame(pd.read_csv(project_scores_file))
    
    # Get project columns
    project_cols = [col for col in df.columns if col.startswith('reward_')]
    project_names = [col.replace('reward_', '') for col in project_cols]
    
    results = {
        'summary': {},
        'best_per_project': {},
        'worst_per_project': {},
        'mean_per_project': {},
        'std_per_project': {},
    }
    
    for col, name in zip(project_cols, project_names):
        results['best_per_project'][name] = df[col].min()  # Lower is better
        results['worst_per_project'][name] = df[col].max()
        results['mean_per_project'][name] = df[col].mean()
        results['std_per_project'][name] = df[col].std()
    
    # Overall statistics
    results['summary']['total_evaluations'] = len(df)
    results['summary']['best_aggregated'] = df['aggregated_reward'].min()
    results['summary']['mean_aggregated'] = df['aggregated_reward'].mean()
    
    return results


def print_project_analysis(results):
    """Pretty print project analysis results"""
    print("\n" + "="*80)
    print("PROJECT PERFORMANCE ANALYSIS")
    print("="*80)
    
    print("\nOVERALL STATISTICS:")
    print(f"  Total evaluations: {results['summary']['total_evaluations']}")
    print(f"  Best aggregated reward: {results['summary']['best_aggregated']:.4f}")
    print(f"  Mean aggregated reward: {results['summary']['mean_aggregated']:.4f}")
    
    print("\nPER-PROJECT PERFORMANCE (Lower is better):")
    print(f"{'Project':<15} {'Best':<10} {'Worst':<10} {'Mean':<10} {'Std':<10}")
    print("-" * 80)
    
    for project in results['best_per_project'].keys():
        print(f"{project:<15} "
              f"{results['best_per_project'][project]:<10.4f} "
              f"{results['worst_per_project'][project]:<10.4f} "
              f"{results['mean_per_project'][project]:<10.4f} "
              f"{results['std_per_project'][project]:<10.4f}")
    
    print("="*80 + "\n")


# Example usage snippet
"""
# In your notebook, replace the reward function setup with:

# 1. Create the tracker
tracker = ProjectScoreTracker(PROJECT_FOLDERS)

# 2. Create enhanced reward function
enhanced_reward_fn = create_enhanced_reward_function(
    project_manager, 
    tracker, 
    aggregation='mean'
)

# 3. Create enhanced collector
collector = EnhancedDataCollector(
    save_dir='rl_analysis_data',
    project_tracker=tracker
)

# 4. Use enhanced reward function in your training
trainner.set_environments(enhanced_reward_fn)

# 5. During training, you can check current project scores
print(tracker.get_current_project_scores())

# 6. After training, analyze results
results = analyze_project_performance('rl_analysis_data/project_scores_*.csv')
print_project_analysis(results)
"""