import wandb
from plato.config import Config
import logging
from datetime import datetime

class WANDBLogger:
    _projectName = Config().data.datasource
    _groupName = Config().params["experiment_name"]
    _entityName = "master-thesis-22"
    
    def configure(cls, projectName, groupName, entityName) -> None:
        cls._projectName = projectName
        cls._groupName = groupName
        cls._entityName = entityName
        
    
    def __init__(self, runName):
        self._initiated: bool = False
        self.runName = self._groupName + "-{}".format(datetime.now().strftime("%d/%m-%H:%M"))
        
        
    def start(self):
        wandb.init(
            project=WANDBLogger._projectName, 
            group=WANDBLogger._groupName, 
            entity=WANDBLogger._entityName
        )
        wandb.run.name = self.runName
        logging.info(
            "[{}] initiated wandb logging for experiment <{}>".format(self.runName, self._groupName)
        )
        self._initiated = True
        
    def define_metric(self, metric, step_metric) -> None:
        wandb.define_metric(step_metric)
        wandb.define_metric(metric, step_metric=step_metric)

    def finish(self) -> None:
        wandb.finish()
        self._initiated = False
    
    
    def log(self,
        data: any,
        step: any = None,
        commit: any = None,
        sync: any = None
    ) -> None:
        if not self._initiated: self.start()
        wandb.log(data, step, commit, sync)