"""
Model Registry - Central storage for all trained models

Tracks all models, their metrics, and automatically selects the best one.

Usage:
    # Register model after training
    python train_with_args.py --register-model
    
    # List all models
    python model_registry.py list
    
    # Deploy best model
    python model_registry.py deploy --model-id model_20260420_143022
    
    # Get production model path
    python model_registry.py production
"""

import json
import os
from datetime import datetime
import sys
import io

# Fix console encoding for Windows
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except Exception:
        pass
from pathlib import Path
import shutil
import argparse


def safe_print(text):
    """Prints text, stripping emojis if encoding fails."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', 'ignore').decode('ascii'))


class ModelRegistry:
    """
    Central registry for tracking and managing trained models.
    
    Features:
    - Track all trained models with metrics
    - Automatically identify best model
    - Deploy models to production
    - Backup previous models
    - Support auto-deployment
    """
    
    def __init__(self, registry_dir='model_registry', keep_only_best=True):
        """
        Initialize model registry.
        
        Args:
            registry_dir: Directory to store registry data
            keep_only_best: If True, only keep the best model (saves disk space)
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(exist_ok=True)
        
        self.registry_file = self.registry_dir / 'registry.json'
        self.models_dir = self.registry_dir / 'models'
        self.models_dir.mkdir(exist_ok=True)
        
        self.keep_only_best = keep_only_best
        
        self.registry = self._load_registry()
    
    def _load_registry(self):
        """Load registry from file."""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                data = json.load(f)
                # Ensure keep_only_best setting is loaded
                if 'keep_only_best' not in data:
                    data['keep_only_best'] = self.keep_only_best
                return data
        return {
            'models': [],
            'production_model': None,
            'best_model': None,
            'keep_only_best': self.keep_only_best,
            'created_at': datetime.now().isoformat()
        }
    
    def _save_registry(self):
        """Save registry to file."""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def register_model(self, model_path, metrics, metadata):
        """
        Register a new trained model.
        
        Args:
            model_path: Path to model checkpoint directory
            metrics: Dict of evaluation metrics (macro_f1, micro_f1, etc.)
            metadata: Dict of training metadata (person, config, etc.)
            
        Returns:
            str: Model ID
        """
        model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Check if new model is better than current best
        current_best = self.get_best_model()
        is_better = False
        
        if current_best is None:
            # First model
            is_better = True
            print(f"\n{'='*70}")
            print(f"📝 FIRST MODEL - Will be registered")
            print(f"{'='*70}")
        else:
            # Compare with current best
            current_best_f1 = current_best['metrics'].get('macro_f1', 0)
            new_f1 = metrics.get('macro_f1', 0)
            
            print(f"\n{'='*70}")
            print(f"📊 COMPARING WITH CURRENT BEST MODEL")
            print(f"{'='*70}")
            print(f"Current Best: {current_best['model_id']}")
            print(f"  Macro F1: {current_best_f1:.4f}")
            print(f"\nNew Model:")
            print(f"  Macro F1: {new_f1:.4f}")
            print(f"\nDifference: {new_f1 - current_best_f1:+.4f}")
            
            if new_f1 > current_best_f1:
                is_better = True
                print(f"\n✅ NEW MODEL IS BETTER! Will replace current model.")
            else:
                print(f"\n❌ NEW MODEL IS NOT BETTER. Will not be saved.")
                print(f"{'='*70}\n")
                return None
        
        # Copy model to registry
        model_registry_path = self.models_dir / model_id
        
        if Path(model_path).exists():
            shutil.copytree(model_path, model_registry_path)
        else:
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
        # Create model entry
        model_entry = {
            'model_id': model_id,
            'path': str(model_registry_path),
            'metrics': metrics,
            'metadata': metadata,
            'registered_at': datetime.now().isoformat(),
            'status': 'registered'
        }
        
        # If keep_only_best mode, remove old models
        if self.keep_only_best and current_best is not None:
            print(f"\n🗑️  Removing old model to save space...")
            self._remove_old_models(keep_model_id=model_id)
        
        self.registry['models'].append(model_entry)
        self._save_registry()
        
        print(f"\n{'='*70}")
        print(f"✅ MODEL REGISTERED")
        print(f"{'='*70}")
        print(f"Model ID: {model_id}")
        print(f"Macro F1: {metrics.get('macro_f1', 'N/A'):.4f}")
        print(f"Micro F1: {metrics.get('micro_f1', 'N/A'):.4f}")
        print(f"Person: {metadata.get('person', 'Unknown')}")
        if self.keep_only_best:
            print(f"\n💾 Storage Mode: Keep Only Best (saves disk space)")
            print(f"   Old models have been removed")
        print(f"{'='*70}\n")
        
        # Auto-evaluate if this is the best model
        self._auto_evaluate()
        
        return model_id
    
    def _remove_old_models(self, keep_model_id=None):
        """
        Remove old models to save disk space.
        
        Args:
            keep_model_id: Model ID to keep (don't delete)
        """
        models_to_remove = []
        
        for model in self.registry['models']:
            if model['model_id'] != keep_model_id:
                models_to_remove.append(model)
        
        for model in models_to_remove:
            # Remove model files
            model_path = Path(model['path'])
            if model_path.exists():
                try:
                    shutil.rmtree(model_path)
                    print(f"   ✓ Removed: {model['model_id']}")
                except Exception as e:
                    print(f"   ⚠️  Could not remove {model['model_id']}: {e}")
            
            # Remove from registry
            self.registry['models'].remove(model)
        
        # Update best_model and production_model if they were removed
        if self.registry.get('best_model') not in [keep_model_id]:
            self.registry['best_model'] = keep_model_id
        
        if self.registry.get('production_model') not in [keep_model_id, None]:
            self.registry['production_model'] = None
        
        self._save_registry()
    
    def _auto_evaluate(self):
        """
        Automatically evaluate and select the best model.
        """
        if not self.registry['models']:
            return
        
        # Find best model by macro F1
        best_model = max(
            self.registry['models'],
            key=lambda m: m['metrics'].get('macro_f1', 0)
        )
        
        current_best_id = self.registry.get('best_model')
        
        if current_best_id != best_model['model_id']:
            print(f"\n{'='*70}")
            print(f"🎉 NEW BEST MODEL FOUND!")
            print(f"{'='*70}")
            print(f"Previous Best: {current_best_id or 'None'}")
            print(f"New Best: {best_model['model_id']}")
            print(f"Macro F1: {best_model['metrics']['macro_f1']:.4f}")
            print(f"Improvement: +{best_model['metrics']['macro_f1'] - self._get_model_by_id(current_best_id)['metrics'].get('macro_f1', 0):.4f}" if current_best_id else "First model")
            print(f"{'='*70}\n")
            
            self.registry['best_model'] = best_model['model_id']
            self._save_registry()
            
            # Trigger auto-deployment if enabled
            if os.getenv('AUTO_DEPLOY', 'false').lower() == 'true':
                print("AUTO_DEPLOY enabled. Deploying best model...")
                self.deploy_model(best_model['model_id'])
    
    def deploy_model(self, model_id):
        """
        Deploy a model to production.
        
        Args:
            model_id: ID of model to deploy
        """
        model = self._get_model_by_id(model_id)
        if not model:
            print(f"✗ Model {model_id} not found in registry")
            return False
        
        print(f"\n{'='*70}")
        print(f"DEPLOYING MODEL TO PRODUCTION")
        print(f"{'='*70}")
        
        # Production directory
        production_dir = Path('saved_model')
        production_dir.mkdir(exist_ok=True)
        
        # Backup current production model
        if production_dir.exists() and any(production_dir.iterdir()):
            backup_dir = self.registry_dir / 'backups' / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_dir.parent.mkdir(exist_ok=True)
            shutil.copytree(production_dir, backup_dir)
            print(f"✓ Backed up current model to: {backup_dir}")
        
        # Clear production directory
        for item in production_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        
        # Deploy new model
        shutil.copytree(model['path'], production_dir, dirs_exist_ok=True)
        
        # Update registry
        # Mark previous production model as registered
        if self.registry['production_model']:
            prev_model = self._get_model_by_id(self.registry['production_model'])
            if prev_model:
                prev_model['status'] = 'registered'
        
        self.registry['production_model'] = model_id
        model['status'] = 'production'
        model['deployed_at'] = datetime.now().isoformat()
        self._save_registry()
        
        print(f"\n✓ Model deployed successfully!")
        print(f"  Model ID: {model_id}")
        print(f"  Macro F1: {model['metrics']['macro_f1']:.4f}")
        print(f"  Micro F1: {model['metrics']['micro_f1']:.4f}")
        print(f"  Deployed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")
        
        return True
    
    def _get_model_by_id(self, model_id):
        """Get model entry by ID."""
        if not model_id:
            return None
        for model in self.registry['models']:
            if model['model_id'] == model_id:
                return model
        return None
    
    def list_models(self, top_n=10, sort_by='macro_f1'):
        """
        List registered models.
        
        Args:
            top_n: Number of top models to show
            sort_by: Metric to sort by ('macro_f1', 'micro_f1', 'registered_at')
        """
        if not self.registry['models']:
            print("\n⚠ No models registered yet.")
            print("Register a model by training with --register-model flag")
            return
        
        # Sort models
        if sort_by in ['macro_f1', 'micro_f1']:
            sorted_models = sorted(
                self.registry['models'],
                key=lambda m: m['metrics'].get(sort_by, 0),
                reverse=True
            )
        else:
            sorted_models = sorted(
                self.registry['models'],
                key=lambda m: m.get(sort_by, ''),
                reverse=True
            )
        
        print(f"\n{'='*80}")
        print(f"MODEL REGISTRY - Top {min(top_n, len(sorted_models))} Models (sorted by {sort_by})")
        if self.keep_only_best:
            print(f"💾 Storage Mode: KEEP ONLY BEST (saves disk space)")
        print(f"{'='*80}\n")
        
        for i, model in enumerate(sorted_models[:top_n], 1):
            # Status icons
            if model['model_id'] == self.registry['production_model']:
                status_icon = "🚀"
                status_text = "PRODUCTION"
            elif model['model_id'] == self.registry['best_model']:
                status_icon = "⭐"
                status_text = "BEST"
            else:
                status_icon = "📦"
                status_text = "REGISTERED"
            
            print(f"{status_icon} {i}. {model['model_id']} [{status_text}]")
            print(f"   {'─'*76}")
            print(f"   Metrics:")
            print(f"     • Macro F1:      {model['metrics'].get('macro_f1', 'N/A'):.4f}")
            print(f"     • Micro F1:      {model['metrics'].get('micro_f1', 'N/A'):.4f}")
            print(f"     • Test Loss:     {model['metrics'].get('test_loss', 'N/A'):.4f}")
            print(f"     • Hamming Loss:  {model['metrics'].get('hamming_loss', 'N/A'):.4f}")
            
            print(f"   Metadata:")
            print(f"     • Person:        {model['metadata'].get('person', 'Unknown')}")
            print(f"     • Experiment:    {model['metadata'].get('experiment_name', 'N/A')}")
            print(f"     • Learning Rate: {model['metadata'].get('learning_rate', 'N/A')}")
            print(f"     • Batch Size:    {model['metadata'].get('batch_size', 'N/A')}")
            print(f"     • Epochs:        {model['metadata'].get('num_epochs', 'N/A')}")
            
            print(f"   Timestamps:")
            print(f"     • Registered:    {model['registered_at']}")
            if 'deployed_at' in model:
                print(f"     • Deployed:      {model['deployed_at']}")
            
            print()
        
        print(f"{'='*80}")
        print(f"Legend: 🚀 = Production | ⭐ = Best | 📦 = Registered")
        print(f"{'='*80}")
        
        # Summary
        print(f"\nSummary:")
        print(f"  Total models: {len(self.registry['models'])}")
        print(f"  Production model: {self.registry['production_model'] or 'None'}")
        print(f"  Best model: {self.registry['best_model'] or 'None'}")
        print()
    
    def get_production_model(self):
        """Get current production model path."""
        prod_id = self.registry.get('production_model')
        if not prod_id:
            return None
        
        model = self._get_model_by_id(prod_id)
        return model['path'] if model else None
    
    def get_best_model(self):
        """Get best model entry (full object)."""
        best_id = self.registry.get('best_model')
        if not best_id:
            # If no best_model set but have models, return first one
            if self.registry['models']:
                return self.registry['models'][0]
            return None
        
        return self._get_model_by_id(best_id)
    
    def get_best_model_path(self):
        """Get best model path."""
        model = self.get_best_model()
        return model['path'] if model else None
    
    def get_model_info(self, model_id):
        """Get detailed information about a model."""
        model = self._get_model_by_id(model_id)
        if not model:
            print(f"✗ Model {model_id} not found")
            return None
        
        print(f"\n{'='*80}")
        print(f"MODEL INFORMATION: {model_id}")
        print(f"{'='*80}\n")
        
        print(f"Status: {model['status'].upper()}")
        print(f"Path: {model['path']}")
        print()
        
        print(f"Metrics:")
        for key, value in model['metrics'].items():
            print(f"  {key}: {value:.4f}")
        print()
        
        print(f"Metadata:")
        for key, value in model['metadata'].items():
            print(f"  {key}: {value}")
        print()
        
        print(f"Registered at: {model['registered_at']}")
        if 'deployed_at' in model:
            print(f"Deployed at: {model['deployed_at']}")
        
        print(f"\n{'='*80}\n")
        
        return model


# CLI Interface
def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description='Model Registry - Manage trained models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all models
  python model_registry.py list
  
  # List top 5 models
  python model_registry.py list --top 5
  
  # Deploy a specific model
  python model_registry.py deploy --model-id model_20260420_143022
  
  # Get production model path
  python model_registry.py production
  
  # Get best model path
  python model_registry.py best
  
  # Get model info
  python model_registry.py info --model-id model_20260420_143022
        """
    )
    
    parser.add_argument(
        'command',
        choices=['list', 'deploy', 'best', 'production', 'info'],
        help='Command to execute'
    )
    parser.add_argument(
        '--model-id',
        help='Model ID (required for deploy and info commands)'
    )
    parser.add_argument(
        '--top',
        type=int,
        default=10,
        help='Number of top models to show (default: 10)'
    )
    parser.add_argument(
        '--sort-by',
        choices=['macro_f1', 'micro_f1', 'registered_at'],
        default='macro_f1',
        help='Metric to sort by (default: macro_f1)'
    )
    
    args = parser.parse_args()
    
    # Initialize registry
    registry = ModelRegistry()
    
    # Execute command
    if args.command == 'list':
        registry.list_models(top_n=args.top, sort_by=args.sort_by)
    
    elif args.command == 'deploy':
        if not args.model_id:
            print("Error: --model-id required for deploy command")
            print("Usage: python model_registry.py deploy --model-id <model_id>")
            return
        registry.deploy_model(args.model_id)
    
    elif args.command == 'best':
        best_model = registry.get_best_model()
        if best_model:
            print(f"\nBest model: {best_model['model_id']}")
            print(f"Path: {best_model['path']}")
            print(f"Macro F1: {best_model['metrics']['macro_f1']:.4f}")
        else:
            print("\n⚠ No best model found. Register models first.")
    
    elif args.command == 'production':
        prod_path = registry.get_production_model()
        if prod_path:
            print(f"\nProduction model path: {prod_path}")
            prod_id = registry.registry['production_model']
            prod_model = registry._get_model_by_id(prod_id)
            print(f"Macro F1: {prod_model['metrics']['macro_f1']:.4f}")
        else:
            print("\n⚠ No production model deployed yet.")
    
    elif args.command == 'info':
        if not args.model_id:
            print("Error: --model-id required for info command")
            print("Usage: python model_registry.py info --model-id <model_id>")
            return
        registry.get_model_info(args.model_id)


if __name__ == "__main__":
    main()
