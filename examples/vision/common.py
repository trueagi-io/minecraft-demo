import logging

logger = logging.getLogger()

class  BaseLoader:
    def load_checkpoint(self, state_dict: dict, strict=True) -> bool:
        """
        Handles size mismatch

        adapted from
        https://github.com/PyTorchLightning/pytorch-lightning/issues/4690
        """
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    logger.info(f"Skip loading parameter: {k}, "
                                f"required shape: {model_state_dict[k].shape}, "
                                f"loaded shape: {state_dict[k].shape}")
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                logger.info(f"Dropping parameter {k}")
                is_changed = True
        self.load_state_dict(state_dict, strict=strict)
        return is_changed
