# π0.5-KI Joint Query Training Design

## Status and naming

The implementation documented here is the **π0.5-lineage, single-stage Knowledge Insulation joint-training query-MSE variant**. Its canonical identifiers are:

- family / tracking project: `pi05_ki`
- model and config prefix: `pi05_ki_joint_query`
- model class: `PI05KIJointQueryPytorch`
- config class: `Pi05KIJointQueryConfig`

This implementation is not evidence of a separate model generation. It must not be described as a distinct successor architecture.

The relevant paper is [Knowledge Insulation: Rethinking Action Backbone Coupling for Vision-Language-Action Models](https://www.pi.website/download/pi05_KI.pdf). The paper formalizes and extends π0.5 (Section 3), describes joint discrete/autoregressive backbone and continuous flow-expert training (Section 5), and introduces the action-expert-to-backbone stop-gradient together with independent backbone action learning (Section 5.2).

The reserved name `pi05_ki_joint_fast` is for a future implementation that actually matches the paper's FAST action-token objective. It must not alias this query-supervised implementation.

## What the current code implements

`src/openpi/models_pytorch/pi05_ki_joint_query.py` extends the existing PyTorch π0.5 subtask model. It jointly optimizes two action-learning paths:

1. a PaliGemma backbone predicts continuous actions from learned query tokens;
2. a separate Gemma action expert predicts flow-matching velocity.

The backbone sequence is conceptually:

```text
[images, robot state, original task prompt, subtask tokens, learned action queries]
```

The action expert receives noisy actions and time embeddings and cross-attends to the backbone prefix cache. The original task prompt remains in the low-level context; subtask tokens augment rather than replace it.

### Backbone objective

Learned query embeddings contain no ground-truth action values. The backbone produces hidden states at the query positions, and `query_action_head` maps those states directly to action space:

```text
query_prediction = query_action_head(query_hidden)
L_query = MSE(query_prediction, ground_truth_action)
```

There is no learned target projection and no ground-truth action injection into the query-token input. Gradients update the query embeddings, backbone, and query action head.

When subtask supervision is present, the backbone also receives causal autoregressive cross-entropy over subtask text:

```text
L_backbone = beta_text * L_subtask_CE + beta_query * L_query
```

The defaults are `beta_text=1.0` and `beta_query=1.0`.

### Action-expert objective

The action expert uses the existing π0.5 continuous flow-matching path:

```text
x_t = t * noise + (1 - t) * action
u_t = noise - action
L_flow = MSE(expert(x_t, t, prefix_KV), u_t)
L_expert = flow_loss_weight * L_flow
```

The default `flow_loss_weight` is `10.0`.

### Knowledge Insulation

Knowledge Insulation is asymmetric gradient routing, not context hiding:

- with `knowledge_insulation=True`, the expert prefix is built without a retained backbone graph and its key/value tensors are detached before expert training;
- therefore `L_flow` updates action-expert parameters but does not update backbone parameters;
- the backbone still learns actions independently through `L_query`, as required for KI to avoid leaving the backbone action-blind;
- with `knowledge_insulation=False`, the flow path is allowed to contribute gradients to the backbone for the controlled ablation.

The implementation exposes separate backbone and expert loss phases so the KI-on path does not retain two large computation graphs at once.

### KV truncation is independent of KI

`truncate_expert_kv=True` truncates the expert's cross-attention cache at the subtask boundary. Query positions are excluded from the expert prefix.

This is separate from gradient insulation:

- truncation controls which forward context the expert can see;
- detachment controls which parameters receive backward gradients;
- truncation applies in both the KI-on and KI-off configurations.

Because the learned queries contain no ground-truth action input, truncation is an architectural boundary rather than a workaround for teacher-forced action-token leakage.

## Deliberate deviations from the paper recipe

This repository's implementation is inspired by the π0.5 KI mechanism but is not paper-exact. The deviations are part of the canonical name:

| Dimension | KI paper recipe | Current `pi05_ki_joint_query` implementation |
|---|---|---|
| Independent backbone action target | Discrete FAST action tokens | Learned action queries decoded to continuous actions |
| Backbone action loss | Autoregressive token cross-entropy | Query-head mean-squared error |
| Additional language target | Paper recipe does not require this repository's subtask target | Adds causal subtask cross-entropy when annotations are available |
| Action expert | Continuous flow matching | Continuous flow matching |
| Flow weighting | Paper recipe, as reported there | Repository-specific weight of `10.0` by default |
| VLM co-training | Paper training mixture includes VLM/language supervision | Not implemented in this training path |
| Training scope | Paper's complete recipe | Robotics-data joint training in this repository |

Consequently, results from this code must be reported as a **π0.5-KI joint query-MSE variant**, not as reproduction evidence for the paper's FAST-action result.

## Training integration

The registered model string is `pi05_ki_joint_query`. `scripts/train_accelerate.py` constructs `PI05KIJointQueryPytorch` and uses one AdamW optimizer with disjoint backbone and expert parameter groups in the Accelerate/DeepSpeed path. The standalone trainer utilities remain available in `src/openpi/training/pi05_ki_joint_trainer.py` for optimizer, checkpoint, schedule, and metrics contracts.

The training configs live in `src/openpi/training/pi05_ki_joint_query_config.py`. Every config uses:

- `project_name="pi05_ki"`;
- a unique `exp_name` and output directory;
- `pytorch_model_name="pi05_ki_joint_query"`;
- explicit KI and truncation flags;
- the π0.5 base checkpoint as initialization.

The multi-task data configuration explicitly selects the first five canonical B1K tasks and applies episode indices per task. This avoids treating LeRobot `episodes_index` as a global index across tasks.

## Inference behavior

The query-head objective is a training signal for backbone action representation learning. Runtime action sampling continues to use the π0.5 flow expert and its denoising loop. Existing hierarchical subtask generation remains observation-conditioned and is recomputed for each policy call.

This implementation does not introduce:

- a FAST tokenizer or FAST autoregressive action decoding;
- a new policy-family generation;
- VLM co-training batches;
- compatibility aliases for superseded Python module or symbol names.

## Checkpoint compatibility

The rename changes Python module, class, config, and experiment identifiers only. It does not rename any `nn.Module` parameter attributes. Safetensors checkpoints are keyed by the module attribute hierarchy (for example, `query_embeddings` and `query_action_head.weight`), not by the Python source filename or class name. Existing state dictionaries therefore retain their keys and can be loaded through the new class.

Optimizer and training-state metadata should use the new identifiers for new runs. Historical output directories, submitted jobs, and existing tracking runs keep their recorded names and are not migrated.

## Required behavioral contracts

Tests for this implementation cover the following invariants:

1. KI on blocks action-expert flow gradients from the backbone.
2. KI off permits the flow-to-backbone gradient path.
3. Learned query inputs contain no ground-truth actions.
4. Query-head parameters receive gradients from the query MSE objective.
5. KV truncation remains enabled independently of the KI toggle.
6. Loss weighting follows `beta_text`, `beta_query`, and `flow_loss_weight`.
7. Checkpoint save/load, metrics, resume, data split, and task-index behavior remain stable.
8. Hierarchical subtask, VLM2 memory neutrality, and DynamicCache preservation retain their existing contracts.
