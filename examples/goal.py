import logging

class Goal:

    def __init__(self, delegate=None):
        # `delegate` is used to delegate all methods to one child
        # any of them can be redefined to alter behavior
        self._delegate = delegate

    def to_json(self):
        return {'goal': self.__class__.__name__,
                'delegate': 'None' if self._delegate is None else self._delegate.to_json()
               }

    @property
    def delegate(self):
        return self._delegate

    @delegate.setter
    def delegate(self, delegate):
        logging.debug(str(self) + ' new delegate ' + str(delegate))
        self._delegate = delegate

    def update(self):
        if self.delegate is None:
            # `update` should not necessarily be implemented
            pass
        else:
            return self.delegate.update()

    def act(self):
        if self.delegate is None:
            raise NotImplementedError
        return self.delegate.act()

    def stop(self):
        if self.delegate is None:
            raise NotImplementedError
        return self.delegate.stop()

    def finished(self):
        if self.delegate is None:
            raise NotImplementedError
        return self.delegate.finished()

    def cycle(self):
        self.update()
        if self.finished():
            return self.stop(), False
        else:
            return self.act(), True


class RobGoal(Goal):

    def __init__(self, rob, delegate=None):
        super().__init__(delegate)
        self.rob = rob


class GoalNode(Goal):

    def __init__(self, subgoals):
        super().__init__()
        self.subgoals = subgoals

    def to_json(self):
        result = super().to_json()
        result['subgoals'] = [g.to_json() for g in self.subgoals]
        return result

    def __str__(self):
        class_name = str(self.__class__).split('\'')[1]
        result = '(' + class_name
        for g in self.subgoals:
            result += str(g)
        result += ')'
        return result


class CAnd(GoalNode):
    '''
    Concurrent And
    Subgoals should be achieved simultaneously.
    Subgoals are being checked and executed even if finished.
    '''

    def __init__(self, subgoals):
        super().__init__(subgoals)

    def update(self):
        for goal in self.subgoals:
            goal.update()

    def act(self):
        acts = []
        for goal in self.subgoals:
            acts += goal.act()
        return acts

    def stop(self):
        acts = []
        for goal in self.subgoals:
            acts += goal.stop()
        return acts

    def finished(self):
        return all([goal.finished() for goal in self.subgoals])


class SAnd(GoalNode):
    '''
    Sequential And
    Subgoals are executed one by one.
    Finished subgoals are not updated.
    '''
    
    def __init__(self, subgoals):
        super().__init__(subgoals)
        self.current = 0

    def to_json(self):
        result = super().to_json()
        result['current'] = self.current
        return result

    def update(self):
        if not self.finished():
            self.subgoals[self.current].update()

    def act(self):
        if self.finished():
            return []
        goal = self.subgoals[self.current]
        if goal.finished():
            self.current += 1
            return goal.stop()
        return goal.act()

    def stop(self):
        if self.finished():
            return []
        return self.subgoals[self.current].stop()

    def finished(self):
        return self.current >= len(self.subgoals)


class COr(CAnd):
    '''
    Concurrent Or
    Subgoals are executed simultaneously.
    But it is enough to achieve any of the goals.
    '''
    def finished(self):
        return any([goal.finished() for goal in self.subgoals])


class Switcher(RobGoal):

    N_TRIES = 3 # when `delegate` is None, Switcher finishes after `N_TRIES`

    def __init__(self, rob):
        super().__init__(rob)
        self.stopDelegate = False
        self.nTries = Switcher.N_TRIES

    def act(self):
        if self.delegate is None:
            return []
        # We stop `delegate` either forcefully or when it is finished.
        if self.stopDelegate or self.delegate.finished():
            acts = self.delegate.stop()
            self.delegate = None
            self.stopDelegate = False
            return acts
        return self.delegate.act()

    def stop(self):
        return [] if self.delegate is None else self.delegate.stop()

    def finished(self):
        if self.delegate is None:
            self.nTries -= 1
        else:
            self.nTries = Switcher.N_TRIES
        return self.nTries <= 0
