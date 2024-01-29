class testerComposit:

  def __init__(self, testers):
    self.testers=testers

  def test(self,net):
    for t in self.testers:
      t.test(net)
