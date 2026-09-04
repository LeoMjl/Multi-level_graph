from mlg.methods.baselines import (
    AtomicRagMethod,
    BoundedContextMethod,
    PlanAndSolveMethod,
    RawRagMethod,
    TruncatedContextMethod,
)
from mlg.methods.ours import (
    OursFullMethod,
    OursWithoutActivePoolMethod,
    OursWithoutDependencyMethod,
    OursWithoutHierarchyMethod,
    OursWithoutMainlineMethod,
)
from mlg.methods.locomo_official import (
    OfficialBaseContextMethod,
    OfficialIncrementalSummarizationMethod,
    OfficialLongContextMethod,
)

METHODS = {
    "ours_full": OursFullMethod,
    "ours_wo_hier": OursWithoutHierarchyMethod,
    "ours_wo_mainline": OursWithoutMainlineMethod,
    "ours_wo_dep": OursWithoutDependencyMethod,
    "ours_wo_active_pool": OursWithoutActivePoolMethod,
    "truncated_context": TruncatedContextMethod,
    "bounded_context": BoundedContextMethod,
    "raw_rag": RawRagMethod,
    "atomic_rag": AtomicRagMethod,
    "plan_and_solve": PlanAndSolveMethod,
    "official_base": OfficialBaseContextMethod,
    "official_long_context": OfficialLongContextMethod,
    "official_incremental": OfficialIncrementalSummarizationMethod,
}

__all__ = [
    "METHODS",
    "OursFullMethod",
    "OursWithoutHierarchyMethod",
    "OursWithoutMainlineMethod",
    "OursWithoutDependencyMethod",
    "OursWithoutActivePoolMethod",
    "TruncatedContextMethod",
    "BoundedContextMethod",
    "RawRagMethod",
    "AtomicRagMethod",
    "PlanAndSolveMethod",
    "OfficialBaseContextMethod",
    "OfficialLongContextMethod",
    "OfficialIncrementalSummarizationMethod",
]
