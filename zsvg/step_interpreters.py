import numpy as np
import torch

from zsvg.util import parse_step
from zsvg.registry import register_interpreter

@register_interpreter
class ClosestInterpreter():
    step_name = ['CLOSEST', 'NEAR', 'BY', 'IN']

    def __init__(self):
        pass

    def execute(self, prog_step, inspect=False):

        parse_result = parse_step(prog_step.prog_str)

        targets = parse_result['args']['targets']
        targets = prog_step.state[targets]

        anchor_names = parse_result['args']['anchors']

        if isinstance(anchor_names, list):
            anchors = prog_step.state[anchor_names[0]]
            anchors1 = prog_step.state[anchor_names[1]]
            target = self.predict(targets, anchors, anchors1)
        else:
            anchors = prog_step.state[anchor_names]
            target = self.predict(targets, anchors)

        output_var = parse_result['output_var']
        prog_step.state[output_var] = target
        return target

    def predict(self, targets, anchors, anchors1=None):
        distances = []

        for i, target in enumerate(targets):
            target_loc = target['obj_loc'][:3]
            dists = []

            for anchor in anchors:
                anchor_loc = anchor['obj_loc'][:3]
                dist = np.linalg.norm(target_loc - anchor_loc)
                if anchors1 is not None:
                    for anchor1 in anchors1:
                        anchor_loc = anchor1['obj_loc'][:3]
                        dist1 = np.linalg.norm(target_loc - anchor_loc)
                        dists.append(dist + dist1)
                else:
                    dists.append(dist)

            distances.append(min(dists))

        targets = [targets[i] for i in np.argsort(distances)]

        return targets

@register_interpreter
class FarthestInterpreter():
    step_name = ['FARTHEST', 'OPPOSITE']

    def __init__(self):
        pass

    def execute(self, prog_step, inspect=False):

        parse_result = parse_step(prog_step.prog_str)

        targets = parse_result['args']['targets']
        targets = prog_step.state[targets]

        anchor_names = parse_result['args']['anchors']

        if isinstance(anchor_names, list):
            anchors = prog_step.state[anchor_names[0]]
            anchors1 = prog_step.state[anchor_names[1]]
            target = self.predict(targets, anchors, anchors1)
        else:
            anchors = prog_step.state[anchor_names]
            target = self.predict(targets, anchors)

        output_var = parse_result['output_var']
        prog_step.state[output_var] = target
        return target

    def predict(self, targets, anchors, anchors1=None):
        distances = []

        for i, target in enumerate(targets):
            target_loc = target['obj_loc'][:3]
            dists = []

            for anchor in anchors:
                anchor_loc = anchor['obj_loc'][:3]
                dist = np.linalg.norm(target_loc - anchor_loc)
                if anchors1 is not None:
                    for anchor1 in anchors1:
                        anchor_loc = anchor1['obj_loc'][:3]
                        dist1 = np.linalg.norm(target_loc - anchor_loc)
                        dists.append(dist + dist1)
                else:
                    dists.append(dist)

            distances.append(max(dists))

        targets = [targets[i] for i in np.flip(np.argsort(distances))]

        return targets



@register_interpreter
class LowerInterpreter():
    step_name = ['LOWER', 'UNDER', 'BELOW']

    def __init__(self) -> None:
        pass

    def execute(self, prog_step, inspect=False):
        parse_result = parse_step(prog_step.prog_str)

        targets = parse_result['args']['targets']
        targets = prog_step.state[targets]

        anchor_names = parse_result['args']['anchors']
        if isinstance(anchor_names, list):
            anchors = prog_step.state[anchor_names[0]]
            anchors1 = prog_step.state[anchor_names[1]]
            anchors = anchors + anchors1
        else:
            anchors = prog_step.state[anchor_names]

        target_id = self.predict(targets, anchors)
        output_var = parse_result['output_var']
        prog_step.state[output_var] = target_id
        return target_id

    def predict(self, targets, anchors):

        result = []
        ids = []

        for i, anchor in enumerate(anchors):
            anchor_height = anchor['obj_loc'][2]
            for j, target in enumerate(targets):
                target_height = target['obj_loc'][2]
                if target_height < anchor_height:
                    ids.append(j)

        ids = list(set(ids))
        result = [targets[i] for i in ids]

        return result

@register_interpreter
class HigherInterpreter():
    step_name = ['HIGHER', 'ABOVE']

    def __init__(self) -> None:
        pass

    def execute(self, prog_step, inspect=False):
        parse_result = parse_step(prog_step.prog_str)

        targets = parse_result['args']['targets']
        targets = prog_step.state[targets]

        anchor_names = parse_result['args']['anchors']
        if isinstance(anchor_names, list):
            anchors = prog_step.state[anchor_names[0]]
            anchors1 = prog_step.state[anchor_names[1]]
            anchors = anchors + anchors1
        else:
            anchors = prog_step.state[anchor_names]

        target_id = self.predict(targets, anchors)
        output_var = parse_result['output_var']
        prog_step.state[output_var] = target_id
        return target_id

    def predict(self, targets, anchors):
        result = []
        ids = []

        for i, anchor in enumerate(anchors):
            anchor_height = anchor['obj_loc'][2]
            for j, target in enumerate(targets):
                target_height = target['obj_loc'][2]
                if target_height > anchor_height:
                    ids.append(j)

        ids = list(set(ids))
        result = [targets[i] for i in ids]

        if len(result) == 0:
            result = targets

        return result

@register_interpreter
class LowestInterpreter():
    step_name = 'LOWEST'

    def __init__(self) -> None:
        pass

    def execute(self, prog_step, inspect=False):
        parse_result = parse_step(prog_step.prog_str)

        step_name = parse_result['step_name']
        assert (step_name == self.step_name)

        targets = parse_result['args']['targets']
        targets = prog_step.state[targets]

        target_id = self.predict(targets)
        output_var = parse_result['output_var']
        prog_step.state[output_var] = target_id
        return target_id

    def predict(self, targets):

        heights = []
        ids = []

        for i, target in enumerate(targets):
            height = target['obj_loc'][2]
            heights.append(height)
            # ids.append(target['obj_id'])

        target = targets[heights.index(min(heights))]

        return [target]

@register_interpreter
class HighestInterpreter():
    step_name = 'HIGHEST'

    def __init__(self) -> None:
        pass

    def execute(self, prog_step, inspect=False):
        parse_result = parse_step(prog_step.prog_str)

        step_name = parse_result['step_name']
        assert (step_name == self.step_name)

        targets = parse_result['args']['targets']
        targets = prog_step.state[targets]

        target_id = self.predict(targets)
        output_var = parse_result['output_var']
        prog_step.state[output_var] = target_id
        return target_id

    def predict(self, targets):

        heights = []
        # ids = []

        for i, target in enumerate(targets):
            height = target['obj_loc'][2]
            heights.append(height)
            # ids.append(target['obj_id'])

        target = targets[heights.index(max(heights))]

        return [target]

@register_interpreter
class CalcSizeInterpreter():
    step_name = 'CALC_SIZE'

    def __init__(self) -> None:
        pass

    def execute(self, prog_step, inspect=False):
        parse_result = parse_step(prog_step.prog_str)

        step_name = parse_result['step_name']
        assert (step_name == self.step_name)

        targets = parse_result['args']['targets']
        targets = prog_step.state[targets]

        target_id = self.predict(targets)

        output_var = parse_result['output_var']
        prog_step.state[output_var] = target_id
        return target_id

    def predict(self, targets):

        sizes = []
        ids = []

        for i, target in enumerate(targets):
            size = np.linalg.norm(target['obj_loc'][3:])
            sizes.append(size)
            ids.append(target['obj_id'])

        return [{'values': sizes, 'ids': ids, 'targets': targets}]

@register_interpreter
class CalcHeightInterpreter():
    step_name = 'CALC_HEIGHT'

    def __init__(self) -> None:
        pass

    def execute(self, prog_step, inspect=False):
        parse_result = parse_step(prog_step.prog_str)

        step_name = parse_result['step_name']
        assert (step_name == self.step_name)

        targets = parse_result['args']['targets']
        targets = prog_step.state[targets]

        target_id = self.predict(targets)

        output_var = parse_result['output_var']
        prog_step.state[output_var] = target_id
        return target_id

    def predict(self, targets):

        heights = []
        ids = []

        for i, target in enumerate(targets):
            height = target['obj_loc'][5]
            heights.append(height)
            ids.append(target['obj_id'])

        return [{'values': heights, 'ids': ids}]

@register_interpreter
class CalcLengthInterpreter():
    step_name = 'CALC_LENGTH'

    def __init__(self) -> None:
        pass

    def execute(self, prog_step, inspect=False):
        parse_result = parse_step(prog_step.prog_str)

        step_name = parse_result['step_name']
        assert (step_name == self.step_name)

        targets = parse_result['args']['targets']
        targets = prog_step.state[targets]

        target_id = self.predict(targets)

        output_var = parse_result['output_var']
        prog_step.state[output_var] = target_id
        return target_id

    def predict(self, targets):

        heights = []
        ids = []

        for i, target in enumerate(targets):
            height = target['obj_loc'][4]
            heights.append(height)
            ids.append(target['obj_id'])

        return [{'values': heights, 'ids': ids}]

@register_interpreter
class CalcDistInterpreter():
    step_name = 'CALC_DIST'

    def __init__(self):
        pass

    def execute(self, prog_step, inspect=False):
        parse_result = parse_step(prog_step.prog_str)

        step_name = parse_result['step_name']
        assert (step_name == self.step_name)

        targets = parse_result['args']['targets']
        anchors = parse_result['args']['anchors']

        targets = prog_step.state[targets]
        anchors = prog_step.state[anchors]

        distances = self.predict(targets, anchors)

        output_var = parse_result['output_var']
        prog_step.state[output_var] = distances
        return distances

    def predict(self, targets, anchors):
        distances = []
        for anchor in anchors:
            anchor_loc = anchor['obj_loc'][:3]
            dists = []
            for target in targets:
                target_loc = target['obj_loc'][:3]
                dist = np.linalg.norm(target_loc - anchor_loc)
                dists.append(dist)
            distances.append({'anchor_id': anchor['obj_id'], 'values': dists, 'targets': targets})

        return distances

@register_interpreter
class CalcZInterpreter():
    step_name = 'CALC_Z'

    def __init__(self):
        pass

    def execute(self, prog_step, inspect=False):
        parse_result = parse_step(prog_step.prog_str)

        step_name = parse_result['step_name']
        assert (step_name == self.step_name)

        targets = parse_result['args']['targets']
        anchors = parse_result['args']['anchors']

        targets = prog_step.state[targets]
        anchors = prog_step.state[anchors]

        distances = self.predict(targets, anchors)

        output_var = parse_result['output_var']
        prog_step.state[output_var] = distances
        return distances

    def predict(self, targets, anchors):
        result = []
        for anchor in anchors:
            anchor_value = anchor['obj_loc'][2]
            dists = []
            target_ids = []
            for target in targets:
                target_value = target['obj_loc'][2]
                dist = anchor_value - target_value
                dists.append(dist)
                target_ids.append(target['obj_id'])
            result.append({'anchor_id': anchor['obj_id'], 'values': dists, 'ids': target_ids})

        return result

@register_interpreter
class CalcYInterpreter():
    step_name = 'CALC_Y'

    def __init__(self):
        pass

    def execute(self, prog_step, inspect=False):
        parse_result = parse_step(prog_step.prog_str)

        step_name = parse_result['step_name']
        assert (step_name == self.step_name)

        projections = parse_result['args']['proj']
        projections = prog_step.state[projections]

        target_id = self.predict(projections)

        output_var = parse_result['output_var']
        prog_step.state[output_var] = target_id
        return target_id

    def predict(self, projections):
        result = []

        for projection in projections:
            anchor_pos = projection['anchor_2d'][0, 0]
            x_dist = []
            ids = []
            for view in projection['view_info']:
                x_dist.append(view['2d'][0, 1].item())
                ids.append(view['obj_id'])

            result.append({'values': x_dist, 'ids': ids})

        return result

@register_interpreter
class MinInterpreter():
    step_name = 'MIN'

    def __init__(self):
        pass

    def execute(self, prog_step, inspect=False):
        parse_result = parse_step(prog_step.prog_str)

        step_name = parse_result['step_name']
        assert (step_name == self.step_name)

        name = parse_result['args']['data']
        data = prog_step.state[name]

        # ic(name, data)

        target_id = self.predict(data)

        output_var = parse_result['output_var']
        prog_step.state[output_var] = target_id
        return target_id

    def predict(self, data):
        ids = []
        for i, item in enumerate(data):
            index = np.argmin(item['values'])
            ids.append(index)

        index = max(ids, key=ids.count)
        target = item['targets'][index]

        return [target]

@register_interpreter
class MaxInterpreter():
    step_name = 'MAX'

    def __init__(self):
        pass

    def execute(self, prog_step, inspect=False):
        parse_result = parse_step(prog_step.prog_str)

        step_name = parse_result['step_name']
        assert (step_name == self.step_name)

        name = parse_result['args']['data']
        data = prog_step.state[name]

        target_id = self.predict(data)

        output_var = parse_result['output_var']
        prog_step.state[output_var] = target_id
        return target_id

    def predict(self, data):
        ids = []
        for i, item in enumerate(data):
            index = np.argmax(item['values'])
            ids.append(index)

        index = max(ids, key=ids.count)
        target = item['targets'][index]

        return [target]
