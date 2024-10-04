from ray.rllib.core import Columns
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


SHARED_ENCODER_ID = "shared_encoder"


class VPGTorchRLModuleUsingSharedEncoder(TorchRLModule):
    """A VPG (vanilla pol. gradient)-style RLModule using a shared encoder."""

    @override(TorchRLModule)
    def setup(self):
        super().setup()

        # Incoming feature dim from the shared encoder.
        feature_dim = self.model_config["feature_dim"]
        hidden_dim = self.model_config["hidden_dim"]

        self._pi_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.action_space.n),
        )

    @override(RLModule)
    def _forward_inference(self, batch):
        with torch.no_grad():
            return self._common_forward(batch)

    @override(RLModule)
    def _forward_exploration(self, batch):
        with torch.no_grad():
            return self._common_forward(batch)

    @override(RLModule)
    def _forward_train(self, batch):
        return self._common_forward(batch)

    def _common_forward(self, batch):
        # Features can be found in the batch under the "encoder_features" key.
        features = batch["encoder_features"]
        logits = self._pi_head(features)
        return {Columns.ACTION_DIST_INPUTS: logits}


class VPGTorchMultiRLModuleWithSharedEncoder(MultiRLModule):
    """A VPG (vanilla policy gradient)-style MultiRLModule with shared encoder.

    This MultiRLModule needs to be configured appropriately as follows:

    .. testcode::
        :skipif: true

        from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
        from ray.rllib.core.rl_module.rl_module import RLModuleSpec

        FEATURE_DIM = 64  # encoder output (feature) dim
        HIDDEN_DIM = 64  # hidden dim for the policy nets

        config.rl_module(
            rl_module_spec=MultiRLModuleSpec(
                module_specs={
                    # Central/shared encoder net.
                    SHARED_ENCODER_ID: RLModuleSpec(
                        module_class=SharedTorchEncoder,
                        model_config_dict={"feature_dim": FEATURE_DIM},
                    ),
                    # Arbitrary number of policy nets (w/o encoder sub-net).
                    "p0": RLModuleSpec(
                        module_class=VPGTorchRLModuleUsingSharedEncoder,
                        model_config_dict={
                            "feature_dim": FEATURE_DIM,
                            "hidden_dim": HIDDEN_DIM,
                        },
                    ),
                    "p1": RLModuleSpec(
                        module_class=VPGTorchRLModuleUsingSharedEncoder,
                        model_config_dict={
                            "feature_dim": FEATURE_DIM,
                            "hidden_dim": HIDDEN_DIM,
                        },
                    ),
                },
            ),
        )

    Also note that in order to learn properly, a special, multi-agent Learner that
    accounts for the shared encoder must be setup. This Learner should have only a
    single optimizer (for all submodules: encoder and all policy nets) in order to not
    destabilize learning. The latter would happen if more than one optimizer would try
    to optimize the same shared encoder submodule.
    """

    @override(MultiRLModule)
    def setup(self):
        super().setup()

        # Assert, we have the shared encoder submodule.
        assert (
            SHARED_ENCODER_ID in self._rl_modules
            and isinstance(self._rl_modules[SHARED_ENCODER_ID], SharedTorchEncoder)
            and len(self._rl_modules) > 1
        )

    @override(MultiRLModule)
    def _run_forward_pass(self, forward_fn_name, batch, **kwargs):
        outputs = {}
        encoder_forward_fn = getattr(
            self._rl_modules[SHARED_ENCODER_ID], forward_fn_name
        )

        for policy_id in batch.keys():
            self._check_module_exists(policy_id)
            rl_module = self._rl_modules[policy_id]
            forward_fn = getattr(rl_module, forward_fn_name)

            # Pass policy's observations through shared encoder to get the features for
            # this policy.
            features = encoder_forward_fn(batch[policy_id])
            # Pass the policy's features through the policy net.
            batch[policy_id]["encoder_features"] = features
            outputs[policy_id] = forward_fn(batch[policy_id], **kwargs)

        return outputs


class SharedTorchEncoder(TorchRLModule):
    """A shared encoder that can be used with VPGTorchRLModuleUsingSharedEncoder."""

    @override(TorchRLModule)
    def setup(self):
        super().setup()

        input_dim = self.observation_space.shape[0]
        feature_dim = self.model_config["feature_dim"]

        self._encoder = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
        )

    @override(RLModule)
    def _forward_inference(self, batch):
        with torch.no_grad():
            return self._common_forward(batch)

    @override(RLModule)
    def _forward_exploration(self, batch):
        with torch.no_grad():
            return self._common_forward(batch)

    @override(RLModule)
    def _forward_train(self, batch):
        return self._common_forward(batch)

    def _common_forward(self, batch):
        # Pass observations through the encoder and return outputs.
        features = self._encoder(batch[Columns.OBS])
        return {"encoder_features": features}