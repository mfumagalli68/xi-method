class XiError(Exception):
    """Generic xi_method exception"""

    def __init__(self, msg):
        self.msg = msg

    def __repr__(self):
        print(self.msg)
