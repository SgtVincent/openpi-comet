# Skill Metric Registry Table

- 覆盖 skill 数: `34`
- 说明: `metric family` 表示评测族；`object roles` 表示从 annotation 中抽取的参数位；`success rule` 是运行时判定语义。

| Skill | Metric Family | Object Roles | Success Rule | Example |
|---|---|---|---|---|
| move to | geometry_base_target | target_or_obj | robot base stays within a target-relative proximity threshold captured from demo end state | task-0000/episode_00000010.json#skill_000 |
| pick up from | grasp_relation | obj, src_or_target | object is grasped and no longer ontop/inside original source | task-0000/episode_00000010.json#skill_001 |
| place in | relation_place_inside | obj, dst_or_target | object is inside destination and released | task-0001/episode_00010010.json#skill_004 |
| place on | relation_place_ontop | obj, dst_or_target | object is ontop destination and released | task-0000/episode_00000010.json#skill_003 |
| push to | geometry_object_target | obj, target_or_dst | object reaches demo-end target pose or becomes nextto target | task-0003/episode_00032920.json#skill_006 |
| chop | contact_effect_proxy | obj, target_obj | tool contacts target object during chopping window | task-0041/episode_00410030.json#skill_015 |
| open door | articulation_open | unary_target | target articulated object is open | task-0003/episode_00030010.json#skill_001 |
| place on next to | relation_place_ontop_nextto | obj, support_target, neighbor_target | object is ontop support, nextto neighbor, and released | task-0002/episode_00020010.json#skill_026 |
| close door | articulation_close | unary_target | target articulated object is closed | task-0003/episode_00030010.json#skill_011 |
| sweep surface | contact_effect_proxy | obj, target_obj_or_surface | tool maintains contact with target surface during sweep | task-0036/episode_00360010.json#skill_004 |
| pour | relation_transfer_proxy | payload_or_obj, dst_or_target, obj | payload reaches target/support or manipulated container reaches end pose proxy | task-0009/episode_00090010.json#skill_003 |
| turn on switch | toggle_on | unary_target | target is toggled on | task-0030/episode_00300010.json#skill_011 |
| close lid | articulation_close | unary_target | lid/container is closed | task-0004/episode_00040010.json#skill_011 |
| turn to | geometry_base_facing | face_target | robot base yaw faces target object within threshold from demo end state | task-0004/episode_00041640.json#skill_003 |
| turn off switch | toggle_off | unary_target | target is toggled off | task-0030/episode_00300010.json#skill_016 |
| hand over | transfer_pose_proxy | obj | object reaches demo-end handover pose while remaining grasped | task-0004/episode_00040010.json#skill_004 |
| spray | contact_effect_proxy | obj, target_obj | sprayer reaches/contacts the target object during the segment | task-0038/episode_00380010.json#skill_004 |
| open lid | articulation_open | unary_target | lid/container is open | task-0004/episode_00040010.json#skill_005 |
| hold | grasp_hold | obj | object is grasped | task-0035/episode_00350010.json#skill_003 |
| release | grasp_release | obj | object is no longer grasped | task-0035/episode_00350010.json#skill_005 |
| tip over | orientation_proxy | obj | object orientation matches tipped end-state proxy | task-0014/episode_00140010.json#skill_016 |
| insert | relation_place_inside | obj, dst_or_target | object is inside destination and released | task-0029/episode_00290010.json#skill_020 |
| sweep off | relation_detach_surface | obj, src_or_target | object is no longer ontop the original surface | task-0044/episode_00440010.json#skill_007 |
| open drawer | articulation_open | unary_target | drawer/openable target is open | task-0002/episode_00020010.json#skill_001 |
| close drawer | articulation_close | unary_target | drawer/openable target is closed | task-0002/episode_00020010.json#skill_013 |
| place in next to | relation_place_inside_nextto | obj, support_target, neighbor_target | object is inside container, nextto neighbor, and released | task-0008/episode_00080010.json#skill_011 |
| place under | relation_under | obj, dst_or_target | object is under the target and released | task-0027/episode_00270010.json#skill_007 |
| pull tray | articulation_open_proxy | unary_target | tray-bearing object is open (pulled out) | task-0049/episode_00490010.json#skill_060 |
| press | toggle_on | unary_target | pressed target is toggled on | task-0000/episode_00000010.json#skill_002 |
| ignite | effect_on_fire | target_obj | target object is on fire | task-0030/episode_00300010.json#skill_014 |
| hang | relation_attach | obj, dst_or_target | object is attached to hanging target and released | task-0034/episode_00340020.json#skill_003 |
| attach | relation_attach | obj, dst_or_target | object is attached to target and released | task-0035/episode_00350010.json#skill_004 |
| wipe hard | contact_effect_proxy | obj, target_obj | cleaning tool contacts target object during wiping | task-0037/episode_00370010.json#skill_004 |
| push tray | articulation_close_proxy | unary_target | tray-bearing object is closed (pushed in) | task-0049/episode_00490010.json#skill_063 |
