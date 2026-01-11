#!/usr/bin/env python3
"""
Tests for Workspace Evaluator

Basic unit tests to validate evaluation framework functionality.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ops.evaluation.workspace_evaluator import WorkspaceEvaluator


class TestWorkspaceEvaluator:
    """Test suite for WorkspaceEvaluator"""
    
    @patch('ops.evaluation.workspace_evaluator.redis.Redis')
    def test_initialization(self, mock_redis):
        """Test evaluator initialization"""
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_redis.return_value = mock_client
        
        evaluator = WorkspaceEvaluator()
        
        assert evaluator.stream_key == 'quantum:stream:trade.intent'
        assert evaluator.min_events == 200
        assert evaluator.cutover_ts is None
        mock_client.ping.assert_called_once()
    
    @patch('ops.evaluation.workspace_evaluator.redis.Redis')
    def test_initialization_with_cutover(self, mock_redis):
        """Test evaluator initialization with cutover timestamp"""
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_redis.return_value = mock_client
        
        cutover_ts = "2026-01-10T05:43:15Z"
        evaluator = WorkspaceEvaluator(cutover_ts=cutover_ts)
        
        assert evaluator.cutover_ts == cutover_ts
    
    @patch('ops.evaluation.workspace_evaluator.redis.Redis')
    def test_redis_connection_failure(self, mock_redis):
        """Test handling of Redis connection failure"""
        mock_redis.side_effect = Exception("Connection refused")
        
        with pytest.raises(RuntimeError, match="Failed to connect to Redis"):
            WorkspaceEvaluator()
    
    def test_calculate_agreement_insufficient_models(self):
        """Test agreement calculation with insufficient models"""
        evaluator = WorkspaceEvaluator.__new__(WorkspaceEvaluator)
        
        model_results = {
            'model1': {
                'analysis': {
                    'action_pcts': {'BUY': 40, 'SELL': 30, 'HOLD': 30}
                }
            }
        }
        active_models = ['model1']
        
        result = evaluator._calculate_agreement(model_results, active_models)
        
        assert result['agreement_pct'] == 0
        assert result['hard_disagree_pct'] == 0
    
    def test_calculate_agreement_healthy(self):
        """Test agreement calculation with healthy ensemble"""
        evaluator = WorkspaceEvaluator.__new__(WorkspaceEvaluator)
        
        model_results = {
            'model1': {
                'analysis': {
                    'action_pcts': {'BUY': 45, 'SELL': 25, 'HOLD': 30}
                }
            },
            'model2': {
                'analysis': {
                    'action_pcts': {'BUY': 40, 'SELL': 30, 'HOLD': 30}
                }
            },
            'model3': {
                'analysis': {
                    'action_pcts': {'BUY': 50, 'SELL': 20, 'HOLD': 30}
                }
            }
        }
        active_models = ['model1', 'model2', 'model3']
        
        result = evaluator._calculate_agreement(model_results, active_models)
        
        # Should have reasonable agreement
        assert 0 <= result['agreement_pct'] <= 100
        assert 0 <= result['hard_disagree_pct'] <= 100
        assert 'avg_action_dist' in result
    
    def test_check_degeneracy_constant_confidence(self):
        """Test degeneracy detection for constant confidence"""
        evaluator = WorkspaceEvaluator.__new__(WorkspaceEvaluator)
        
        model_results = {
            'stuck_model': {
                'analysis': {
                    'confidence': {
                        'std': 0.005,  # Very low std
                        'mean': 0.5,
                        'p10': 0.5,
                        'p90': 0.5,
                        'p10_p90_range': 0.0
                    },
                    'action_pcts': {'BUY': 33, 'SELL': 33, 'HOLD': 34},
                    'action_counts': {'BUY': 100, 'SELL': 100, 'HOLD': 100},
                    'confidence_violations': []
                }
            }
        }
        
        result = evaluator._check_degeneracy(model_results)
        
        assert result['has_degeneracy'] is True
        assert result['count'] == 1
        assert len(result['degenerate_models']) == 1
        assert 'Constant confidence' in result['degenerate_models'][0]['reasons'][0]
    
    def test_check_degeneracy_hold_collapse(self):
        """Test degeneracy detection for HOLD collapse"""
        evaluator = WorkspaceEvaluator.__new__(WorkspaceEvaluator)
        
        model_results = {
            'hold_model': {
                'analysis': {
                    'confidence': {
                        'std': 0.1,
                        'mean': 0.5,
                        'p10': 0.4,
                        'p90': 0.6,
                        'p10_p90_range': 0.2
                    },
                    'action_pcts': {'BUY': 5, 'SELL': 3, 'HOLD': 92},  # >90% HOLD
                    'action_counts': {'BUY': 10, 'SELL': 6, 'HOLD': 184},
                    'confidence_violations': []
                }
            }
        }
        
        result = evaluator._check_degeneracy(model_results)
        
        assert result['has_degeneracy'] is True
        assert result['count'] == 1
        assert 'HOLD collapse' in result['degenerate_models'][0]['reasons'][0]
    
    def test_check_degeneracy_single_action_dominance(self):
        """Test degeneracy detection for single action dominance"""
        evaluator = WorkspaceEvaluator.__new__(WorkspaceEvaluator)
        
        model_results = {
            'buy_only_model': {
                'analysis': {
                    'confidence': {
                        'std': 0.1,
                        'mean': 0.7,
                        'p10': 0.6,
                        'p90': 0.8,
                        'p10_p90_range': 0.2
                    },
                    'action_pcts': {'BUY': 97, 'SELL': 2, 'HOLD': 1},  # >95% BUY
                    'action_counts': {'BUY': 194, 'SELL': 4, 'HOLD': 2},
                    'confidence_violations': []
                }
            }
        }
        
        result = evaluator._check_degeneracy(model_results)
        
        assert result['has_degeneracy'] is True
        assert result['count'] == 1
        assert 'BUY dominance' in result['degenerate_models'][0]['reasons'][0]
    
    def test_check_degeneracy_healthy_models(self):
        """Test degeneracy detection with healthy models"""
        evaluator = WorkspaceEvaluator.__new__(WorkspaceEvaluator)
        
        model_results = {
            'healthy_model': {
                'analysis': {
                    'confidence': {
                        'std': 0.15,  # Good variance
                        'mean': 0.6,
                        'p10': 0.45,
                        'p90': 0.75,
                        'p10_p90_range': 0.30  # Good range
                    },
                    'action_pcts': {'BUY': 40, 'SELL': 35, 'HOLD': 25},  # Balanced
                    'action_counts': {'BUY': 80, 'SELL': 70, 'HOLD': 50},
                    'confidence_violations': []
                }
            }
        }
        
        result = evaluator._check_degeneracy(model_results)
        
        assert result['has_degeneracy'] is False
        assert result['count'] == 0
        assert len(result['degenerate_models']) == 0
    
    def test_determine_status_pass(self):
        """Test status determination for passing workspace"""
        evaluator = WorkspaceEvaluator.__new__(WorkspaceEvaluator)
        
        model_results = {
            'model1': {'status': 'PASS', 'failures': []},
            'model2': {'status': 'PASS', 'failures': []},
            'model3': {'status': 'PASS', 'failures': []}
        }
        
        ensemble_health = {
            'status': 'HEALTHY',
            'active_model_count': 3
        }
        
        degeneracy_check = {
            'has_degeneracy': False,
            'count': 0
        }
        
        result = evaluator._determine_status(model_results, ensemble_health, degeneracy_check)
        
        assert result['status'] == 'PASS'
        assert len(result['blockers']) == 0
        assert 'SAFE TO PROCEED' in result['recommendations'][0]
    
    def test_determine_status_fail_with_blockers(self):
        """Test status determination with blockers"""
        evaluator = WorkspaceEvaluator.__new__(WorkspaceEvaluator)
        
        model_results = {
            'model1': {'status': 'FAIL', 'failures': ['Too much HOLD']},
            'model2': {'status': 'PASS', 'failures': []},
            'model3': {'status': 'PASS', 'failures': []}
        }
        
        ensemble_health = {
            'status': 'HEALTHY',
            'active_model_count': 2
        }
        
        degeneracy_check = {
            'has_degeneracy': True,
            'count': 1
        }
        
        result = evaluator._determine_status(model_results, ensemble_health, degeneracy_check)
        
        assert result['status'] == 'FAIL_BLOCKERS'
        assert len(result['blockers']) > 0
        assert 'DO NOT activate' in result['recommendations'][0]
    
    def test_determine_status_pass_with_warnings(self):
        """Test status determination with warnings but no blockers"""
        evaluator = WorkspaceEvaluator.__new__(WorkspaceEvaluator)
        
        model_results = {
            'model1': {'status': 'PASS', 'failures': []},
            'model2': {'status': 'PASS', 'failures': []}
        }
        
        ensemble_health = {
            'status': 'WARNING',
            'active_model_count': 2
        }
        
        degeneracy_check = {
            'has_degeneracy': False,
            'count': 0
        }
        
        result = evaluator._determine_status(model_results, ensemble_health, degeneracy_check)
        
        assert result['status'] == 'PASS_WITH_WARNINGS'
        assert len(result['warnings']) > 0
        assert len(result['blockers']) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
