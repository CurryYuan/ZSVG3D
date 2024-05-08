import numpy as np
import torch

from pytorch3d.renderer import look_at_view_transform, FoVOrthographicCameras

from zsvg.util import parse_step
from zsvg.registry import register_interpreter


def egoview_project(targets, anchors, center):

    projections = []
    for i in range(len(anchors)):

        anchor_obj_loc = torch.tensor(anchors[i]['obj_loc']).cuda().unsqueeze(0)
        # cam_pos = torch.tensor([[0.0, 0.0, 1.0]]).cuda()
        cam_pos = torch.tensor(center).cuda().unsqueeze(0).float()

        R, T = look_at_view_transform(eye=cam_pos, at=anchor_obj_loc[:, :3], up=((0.0, 0.0, 1.0),))
        camera = FoVOrthographicCameras(device='cuda', R=R, T=T)

        # print(f'Observation {i}:')
        view_info = []
        for obj in targets:
            pos = torch.tensor(obj['obj_loc'][:3]).cuda().unsqueeze(0)
            pos = camera.transform_points_screen(pos, image_size=(512, 2048))

            view_info.append({'obj_id': obj['obj_id'], 'obj_name': obj['obj_name'], '2d': pos})

        pos = camera.transform_points_screen(anchor_obj_loc[:, :3], image_size=(512, 2048))

        projections.append({'anchor_2d': pos, 'view_info': view_info})

    return projections


class ViewDepInterpreter():
    def __init__(self):
        pass

    def execute(self, prog_step, inspect=False):

        parse_result = parse_step(prog_step.prog_str)

        step_name = parse_result['step_name']
        assert (step_name == self.step_name)

        targets = parse_result['args']['targets']
        targets = prog_step.state[targets]

        if hasattr(parse_result['args'], 'anchors'):
            anchors = parse_result['args']['anchors']
            anchors = prog_step.state[anchors]
        else:
            anchors = targets

        center = prog_step.state['CENTER'][0]['obj_loc']

        result = self.predict(targets, anchors, center)

        output_var = parse_result['output_var']
        prog_step.state[output_var] = result
        return result

@register_interpreter
class LeftInterpreter(ViewDepInterpreter): 
    step_name = 'LEFT'

    def __init__(self):
        super().__init__()

    def predict(self, targets, anchors, center):
        result = []
        ids = []
        x_values = []

        projections = egoview_project(targets, anchors, center)

        for projection in projections:
            anchor_pos = projection['anchor_2d'][0, 0]

            for i, view in enumerate(projection['view_info']):
                if view['2d'][0, 0] < anchor_pos:
                    ids.append(i)
                    x_values.append(view['2d'][0, 0])

        ids = [x for _, x in sorted(zip(x_values, ids))]
        ids = list(dict.fromkeys(ids))
        result = [targets[i] for i in ids]

        if len(result) == 0:
            result = targets

        return result

@register_interpreter
class RightInterpreter(ViewDepInterpreter):
    step_name = 'RIGHT'

    def __init__(self):
        super().__init__()

    def predict(self, targets, anchors, center):
        result = []
        ids = []
        x_values = []

        projections = egoview_project(targets, anchors, center)

        for projection in projections:
            anchor_pos = projection['anchor_2d'][0, 0]

            for i, view in enumerate(projection['view_info']):
                if view['2d'][0, 0] > anchor_pos:
                    ids.append(i)
                    x_values.append(view['2d'][0, 0])

        ids = [x for _, x in sorted(zip(x_values, ids), reverse=True)]
        ids = list(dict.fromkeys(ids))
        result = [targets[i] for i in ids]

        if len(result) == 0:
            result = targets

        return result


@register_interpreter
class FrontInterpreter(ViewDepInterpreter):
    step_name = 'FRONT'

    def __init__(self):
        super().__init__()

    def predict(self, targets, anchors, center):
        result = []
        ids = []
        x_values = []

        projections = egoview_project(targets, anchors, center)

        for projection in projections:
            anchor_pos = projection['anchor_2d'][0, 2]

            for i, view in enumerate(projection['view_info']):
                if view['2d'][0, 2] < anchor_pos:
                    ids.append(i)
                    x_values.append(view['2d'][0, 2])

        ids = [x for _, x in sorted(zip(x_values, ids), reverse=True)]
        ids = list(dict.fromkeys(ids))
        result = [targets[i] for i in ids]

        if len(result) == 0:
            result = targets

        return result

@register_interpreter
class BehindInterpreter(ViewDepInterpreter):
    step_name = 'BEHIND'

    def __init__(self):
        super().__init__()

    def predict(self, targets, anchors, center):
        result = []
        ids = []
        x_values = []

        projections = egoview_project(targets, anchors, center)

        for projection in projections:
            anchor_pos = projection['anchor_2d'][0, 2]

            for i, view in enumerate(projection['view_info']):
                if view['2d'][0, 2] > anchor_pos:
                    ids.append(i)
                    x_values.append(view['2d'][0, 2])

        ids = [x for _, x in sorted(zip(x_values, ids))]
        ids = list(dict.fromkeys(ids))
        result = [targets[i] for i in ids]

        if len(result) == 0:
            result = targets

        return result

@register_interpreter
class LeftMostInterpreter():
    step_name = 'LEFTMOST'

    def __init__(self):
        pass

    def execute(self, prog_step, inspect=False):

        parse_result = parse_step(prog_step.prog_str)

        step_name = parse_result['step_name']
        assert (step_name == self.step_name)

        targets = parse_result['args']['targets']
        targets = prog_step.state[targets]
        center = prog_step.state['CENTER'][0]['obj_loc']

        target_id = self.predict(targets, center)

        output_var = parse_result['output_var']
        prog_step.state[output_var] = target_id
        return target_id

    def predict(self, targets, center):

        projections = egoview_project(targets, targets, center)

        ids = []
        for projection in projections:
            anchor_pos = projection['anchor_2d'][0, 0]
            x_dist = []
            for view in projection['view_info']:
                x_dist.append(view['2d'][0, 0])

            index = torch.argmin(torch.stack(x_dist))
            # id = projection['view_info'][index]['obj_id']
            ids.append(index)

        index = max(ids, key=ids.count)
        target = targets[index]

        return [target]

@register_interpreter
class RightMostInterpreter():
    step_name = 'RIGHTMOST'

    def __init__(self):
        pass

    def execute(self, prog_step, inspect=False):

        parse_result = parse_step(prog_step.prog_str)

        step_name = parse_result['step_name']
        assert (step_name == self.step_name)

        targets = parse_result['args']['targets']
        targets = prog_step.state[targets]
        center = prog_step.state['CENTER'][0]['obj_loc']

        target_id = self.predict(targets, center)

        output_var = parse_result['output_var']
        prog_step.state[output_var] = target_id
        return target_id

    def predict(self, targets, center):

        projections = egoview_project(targets, targets, center)

        ids = []
        for projection in projections:
            anchor_pos = projection['anchor_2d'][0, 0]
            x_dist = []
            for view in projection['view_info']:
                x_dist.append(view['2d'][0, 0])

            index = torch.argmax(torch.stack(x_dist))
            # id = projection['view_info'][index]['obj_id']
            ids.append(index)

        index = max(ids, key=ids.count)
        target = targets[index]

        return [target]

@register_interpreter
class CalcXInterpreter():
    step_name = 'CALC_X'

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
                x_dist.append(view['2d'][0, 0].item())
                ids.append(view['obj_id'])

            result.append({'values': x_dist, 'ids': ids})

        return result

@register_interpreter
class MidInterpreter():
    step_name = 'MID'

    def __init__(self):
        pass

    def execute(self, prog_step, inspect=False):
        parse_result = parse_step(prog_step.prog_str)

        step_name = parse_result['step_name']
        assert (step_name == self.step_name)

        targets = parse_result['args']['targets']
        targets = prog_step.state[targets]

        # if hasattr(parse_result['args'], 'anchors'):
        #     anchors = parse_result['args']['anchors']
        #     anchors = prog_step.state[anchors]
        #     targets = targets + anchors

        center = prog_step.state['CENTER'][0]['obj_loc']

        projections = egoview_project(targets, targets, center)

        x_value = CalcXInterpreter().predict(projections)

        target_id = self.predict(x_value)

        for target in targets:
            if target['obj_id'] == target_id:
                output_var = parse_result['output_var']
                prog_step.state[output_var] = target
                return [target]

    def predict(self, data):
        ids = []

        for item in data:
            values = np.argsort(item['values'])
            index = values[len(values) // 2]
            ids.append(item['ids'][index])

        return max(ids, key=ids.count)

@register_interpreter
class BetweenInterpreter():
    step_name = 'BETWEEN'

    def __init__(self):
        pass

    def execute(self, prog_step, inspect=False):
        parse_result = parse_step(prog_step.prog_str)

        step_name = parse_result['step_name']
        assert (step_name == self.step_name)

        targets = parse_result['args']['targets']
        targets = prog_step.state[targets]

        anchor_names = parse_result['args']['anchors']

        if isinstance(anchor_names, list):
            anchors1 = prog_step.state[anchor_names[0]]
            anchors2 = prog_step.state[anchor_names[1]]
        else:
            anchors1 = prog_step.state[anchor_names]
            anchors2 = prog_step.state[anchor_names]

        center = prog_step.state['CENTER'][0]['obj_loc']

        target = self.predict(targets, anchors1, anchors2, center)

        output_var = parse_result['output_var']
        prog_step.state[output_var] = target
        return target

    def predict(self, targets, anchors1, anchors2, center):

        final_result = []

        result = LeftInterpreter().predict(targets, anchors1, center)
        if len(result) > 0:
            result = RightInterpreter().predict(result, anchors2, center)
            if len(result) > 0:
                final_result.extend(result)

        result = RightInterpreter().predict(targets, anchors1, center)
        if len(result) > 0:
            result = LeftInterpreter().predict(result, anchors2, center)
            if len(result) > 0:
                final_result.extend(result)

        if len(final_result) == 0:
            final_result = targets

        return final_result