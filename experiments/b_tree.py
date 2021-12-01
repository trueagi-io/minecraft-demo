import logging

logger = logging.getLogger()


class Node:
    def __init__(self, nodes):
        self.nodes = nodes
        self.idx = 0 
        self.finished = False

    def reset(self):
        self.idx = 0
        self.finished = False
        for node in self.nodes:
            node.reset()

    def getLastNode(self):
        result = self.nodes[-1]
        if self.idx < len(self.nodes):
            result = self.nodes[self.idx]
        if isinstance(result, Node):
            result = result.getLastNode()
        return result


class And(Node):
    """
    Evaluates children until the first failure
    """
    def __call__(self):
        """
        If success then run text child
        otherwise return same as child
        """
        assert not self.finished
        node = self.nodes[self.idx]
        status, result = node()
        self.prev_status = status
        if status == 'success':
            self.idx += 1
            if self.idx == len(self.nodes):
                logger.debug("last node {0} succeded, we are done".format(node))
                self.finished = True
                return 'success', result
            logger.debug("{0} succeded, choosing next".format(node))
            return self() 
        if status == 'failed':
           logger.debug("{0} failed, we are done".format(node))
           self.finished = True
        return status, result    


class Or(Node):
    """
    Evaluates children until first success or running
    then returns action
    """
    def __init__(self, nodes):
        super().__init__(nodes)
        self.idx = 0

    def __call__(self):
        """
        If child failed then call next child,
        otherwise return same as child
        """
        assert not self.finished
        if self.idx == len(self.nodes):
            # all failed
            logger.debug('all children failed, we\'re done'.format(node))
            self.finished = True
            return 'failure', None 
        node = self.nodes[self.idx]
        status, result = node()
        if status ==  'success':
            logger.debug('{0} succeded, we\'re done'.format(node))
            self.finished = True
            return status, result

        if status == 'running':
            return status, result
        else:
            logger.debug('{0} failed, selecting next node'.format(node))
            # failed, so run text
            self.idx += 1
            return self()

