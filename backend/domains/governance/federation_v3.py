"""
Governance Domain - Federation v3

Multi-agent coordination and consensus decision-making.

Author: Quantum Trader - Hedge Fund OS v2
Date: December 3, 2025
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class VoteType(Enum):
    """Types of votes."""
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"


class DecisionStatus(Enum):
    """Decision proposal statuses."""
    PROPOSED = "proposed"
    VOTING = "voting"
    APPROVED = "approved"
    REJECTED = "rejected"
    VETOED = "vetoed"


@dataclass
class Vote:
    """Vote cast by an agent."""
    voter_id: str
    vote_type: VoteType
    rationale: str
    cast_at: datetime


@dataclass
class DecisionProposal:
    """Decision proposal requiring consensus."""
    decision_id: str
    proposer: str
    decision_type: str
    description: str
    requires_approval_from: List[str]
    votes: List[Vote]
    status: DecisionStatus
    proposed_at: datetime
    decided_at: Optional[datetime] = None


class FederationV3:
    """
    Federation v3 - Multi-agent coordination and consensus.
    
    Responsibilities:
    - Coordinate decision-making across CEO, CRO, CIO
    - Manage voting and consensus protocols
    - Resolve conflicts between agents
    - Track decision history and rationales
    - Enforce voting quorum and majority rules
    
    Authority: COORDINATOR (facilitates, does not decide)
    """
    
    def __init__(
        self,
        policy_store,
        event_bus,
        voting_timeout_minutes: int = 5,
        quorum_percentage: float = 0.67,  # 67% quorum required
        majority_percentage: float = 0.67,  # 67% majority to approve
    ):
        """
        Initialize Federation v3.
        
        Args:
            policy_store: PolicyStore v2 instance
            event_bus: EventBus v2 instance
            voting_timeout_minutes: Minutes to wait for votes
            quorum_percentage: Percentage of voters required for quorum
            majority_percentage: Percentage required to approve
        """
        self.policy_store = policy_store
        self.event_bus = event_bus
        
        # Voting parameters
        self.voting_timeout_minutes = voting_timeout_minutes
        self.quorum_percentage = quorum_percentage
        self.majority_percentage = majority_percentage
        
        # Decision tracking
        self.active_proposals: Dict[str, DecisionProposal] = {}
        self.decision_history: List[DecisionProposal] = []
        
        # Agent registry
        self.registered_agents: Set[str] = {"CEO", "CRO", "CIO"}
        
        # Subscribe to events
        self.event_bus.subscribe("governance.decision.proposed", self._handle_proposal)
        self.event_bus.subscribe("governance.vote.cast", self._handle_vote)
        self.event_bus.subscribe("fund.risk.veto.issued", self._handle_veto)
        
        logger.info(
            f"[Federation] Federation v3 initialized:\n"
            f"   Voting Timeout: {voting_timeout_minutes} minutes\n"
            f"   Quorum: {quorum_percentage:.0%}\n"
            f"   Majority: {majority_percentage:.0%}\n"
            f"   Registered Agents: {', '.join(self.registered_agents)}"
        )
    
    async def propose_decision(
        self,
        proposer: str,
        decision_type: str,
        description: str,
        requires_approval_from: Optional[List[str]] = None,
        **additional_data
    ) -> str:
        """
        Propose a decision for consensus.
        
        Args:
            proposer: Agent proposing the decision
            decision_type: Type of decision
            description: Decision description
            requires_approval_from: List of agents required to vote (None = all)
            **additional_data: Additional decision data
        
        Returns:
            Decision ID
        """
        import uuid
        
        decision_id = f"FED-DEC-{uuid.uuid4().hex[:8].upper()}"
        
        if requires_approval_from is None:
            requires_approval_from = list(self.registered_agents)
        
        proposal = DecisionProposal(
            decision_id=decision_id,
            proposer=proposer,
            decision_type=decision_type,
            description=description,
            requires_approval_from=requires_approval_from,
            votes=[],
            status=DecisionStatus.VOTING,
            proposed_at=datetime.now(timezone.utc)
        )
        
        self.active_proposals[decision_id] = proposal
        
        logger.info(
            f"[Federation] 📋 Decision proposed: {decision_id}\n"
            f"   Proposer: {proposer}\n"
            f"   Type: {decision_type}\n"
            f"   Description: {description}\n"
            f"   Requires votes from: {', '.join(requires_approval_from)}"
        )
        
        # Publish proposal event
        await self.event_bus.publish(
            "governance.decision.proposed",
            {
                "decision_id": decision_id,
                "proposer": proposer,
                "decision_type": decision_type,
                "description": description,
                "requires_approval_from": requires_approval_from,
                "proposed_at": proposal.proposed_at.isoformat(),
                **additional_data
            }
        )
        
        return decision_id
    
    async def cast_vote(
        self,
        decision_id: str,
        voter_id: str,
        vote_type: VoteType,
        rationale: str
    ) -> bool:
        """
        Cast vote on a decision.
        
        Args:
            decision_id: Decision identifier
            voter_id: Agent casting vote
            vote_type: Type of vote (approve/reject/abstain)
            rationale: Vote rationale
        
        Returns:
            True if vote accepted
        """
        proposal = self.active_proposals.get(decision_id)
        if not proposal:
            logger.error(f"[Federation] Decision {decision_id} not found")
            return False
        
        # Check if voter is authorized
        if voter_id not in proposal.requires_approval_from:
            logger.error(
                f"[Federation] {voter_id} not authorized to vote on {decision_id}"
            )
            return False
        
        # Check if already voted
        if any(v.voter_id == voter_id for v in proposal.votes):
            logger.warning(
                f"[Federation] {voter_id} already voted on {decision_id}"
            )
            return False
        
        # Record vote
        vote = Vote(
            voter_id=voter_id,
            vote_type=vote_type,
            rationale=rationale,
            cast_at=datetime.now(timezone.utc)
        )
        proposal.votes.append(vote)
        
        logger.info(
            f"[Federation] 🗳️ Vote cast on {decision_id}:\n"
            f"   Voter: {voter_id}\n"
            f"   Vote: {vote_type.value}\n"
            f"   Rationale: {rationale}"
        )
        
        # Publish vote event
        await self.event_bus.publish(
            "governance.vote.cast",
            {
                "decision_id": decision_id,
                "voter_id": voter_id,
                "vote_type": vote_type.value,
                "rationale": rationale,
                "cast_at": vote.cast_at.isoformat()
            }
        )
        
        # Check if decision can be finalized
        await self._check_decision_finalization(decision_id)
        
        return True
    
    async def _check_decision_finalization(self, decision_id: str) -> None:
        """Check if decision has enough votes to finalize."""
        proposal = self.active_proposals.get(decision_id)
        if not proposal or proposal.status != DecisionStatus.VOTING:
            return
        
        required_voters = len(proposal.requires_approval_from)
        votes_cast = len(proposal.votes)
        
        # Check quorum
        quorum = int(required_voters * self.quorum_percentage)
        if votes_cast < quorum:
            logger.debug(
                f"[Federation] {decision_id}: {votes_cast}/{required_voters} votes, "
                f"quorum not reached ({quorum} required)"
            )
            return
        
        # Count approve votes
        approve_votes = len([v for v in proposal.votes if v.vote_type == VoteType.APPROVE])
        reject_votes = len([v for v in proposal.votes if v.vote_type == VoteType.REJECT])
        
        # Check majority
        majority_threshold = int(votes_cast * self.majority_percentage)
        
        if approve_votes >= majority_threshold:
            # Decision approved
            proposal.status = DecisionStatus.APPROVED
            proposal.decided_at = datetime.now(timezone.utc)
            
            logger.info(
                f"[Federation] ✅ Decision APPROVED: {decision_id}\n"
                f"   Votes: {approve_votes} approve, {reject_votes} reject\n"
                f"   Majority: {approve_votes}/{votes_cast} ≥ {majority_threshold}"
            )
            
            await self.event_bus.publish(
                "governance.decision.approved",
                {
                    "decision_id": decision_id,
                    "proposer": proposal.proposer,
                    "decision_type": proposal.decision_type,
                    "approve_votes": approve_votes,
                    "reject_votes": reject_votes,
                    "total_votes": votes_cast,
                    "decided_at": proposal.decided_at.isoformat()
                }
            )
            
            # Move to history
            self.decision_history.append(proposal)
            del self.active_proposals[decision_id]
            
        elif reject_votes >= majority_threshold:
            # Decision rejected
            proposal.status = DecisionStatus.REJECTED
            proposal.decided_at = datetime.now(timezone.utc)
            
            logger.warning(
                f"[Federation] ❌ Decision REJECTED: {decision_id}\n"
                f"   Votes: {approve_votes} approve, {reject_votes} reject\n"
                f"   Majority: {reject_votes}/{votes_cast} ≥ {majority_threshold}"
            )
            
            await self.event_bus.publish(
                "governance.decision.rejected",
                {
                    "decision_id": decision_id,
                    "proposer": proposal.proposer,
                    "decision_type": proposal.decision_type,
                    "approve_votes": approve_votes,
                    "reject_votes": reject_votes,
                    "total_votes": votes_cast,
                    "decided_at": proposal.decided_at.isoformat()
                }
            )
            
            # Move to history
            self.decision_history.append(proposal)
            del self.active_proposals[decision_id]
    
    async def _handle_proposal(self, event_data: dict) -> None:
        """Handle decision proposals."""
        # Proposal already handled by propose_decision()
        pass
    
    async def _handle_vote(self, event_data: dict) -> None:
        """Handle vote events."""
        # Vote already handled by cast_vote()
        pass
    
    async def _handle_veto(self, event_data: dict) -> None:
        """Handle CRO veto events."""
        veto_id = event_data.get("veto_id")
        decision_type = event_data.get("decision_type")
        target_entity = event_data.get("target_entity")
        
        logger.critical(
            f"[Federation] 🚫 CRO VETO issued: {veto_id}\n"
            f"   Type: {decision_type}\n"
            f"   Target: {target_entity}"
        )
        
        # Check if any active proposals affected by veto
        for decision_id, proposal in list(self.active_proposals.items()):
            # If veto affects this proposal, mark as vetoed
            # (Simple logic: if veto mentions same target)
            if proposal.status == DecisionStatus.VOTING:
                logger.info(
                    f"[Federation] Checking if veto affects proposal {decision_id}"
                )
                # TODO: Implement veto propagation logic
    
    def get_status(self) -> dict:
        """Get federation status."""
        return {
            "registered_agents": list(self.registered_agents),
            "voting_timeout_minutes": self.voting_timeout_minutes,
            "quorum_percentage": self.quorum_percentage,
            "majority_percentage": self.majority_percentage,
            "active_proposals": len(self.active_proposals),
            "decision_history": len(self.decision_history),
            "proposals_by_status": {
                "voting": len([p for p in self.active_proposals.values() if p.status == DecisionStatus.VOTING]),
                "approved": len([p for p in self.decision_history if p.status == DecisionStatus.APPROVED]),
                "rejected": len([p for p in self.decision_history if p.status == DecisionStatus.REJECTED]),
                "vetoed": len([p for p in self.decision_history if p.status == DecisionStatus.VETOED])
            }
        }
