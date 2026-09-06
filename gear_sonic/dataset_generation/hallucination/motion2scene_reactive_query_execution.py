"""Scene-query-enabled revision of the immutable interface pilot."""

from .motion2scene_reactive_execution import ReactiveExecutionEnvCfg


class QueryEnabledExecutionEnvCfg(ReactiveExecutionEnvCfg):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.sim.enable_scene_query_support = True
