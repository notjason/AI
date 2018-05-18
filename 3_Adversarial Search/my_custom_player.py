
from sample_players import DataPlayer


class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only *required* method. You can modify
    the interface for get_action by adding named parameters with default
    values, but the function MUST remain compatible with the default
    interface.

    **********************************************************************
    NOTES:
    - You should **ONLY** call methods defined on your agent class during
      search; do **NOT** add or call functions outside the player class.
      The isolation library wraps each method of this class to interrupt
      search when the time limit expires, but the wrapper only affects
      methods defined on this class.

    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.
    **********************************************************************
    """
   
    def score(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - len(opp_liberties)


    def iterative(self,gameState,f, depth_limit): 
        best_move = None
        for depth in range(1, depth_limit+1):
            best_move = CustomPlayer.mtdf(self,gameState,f, depth)
        return best_move


    def AlphaBetaWithMemory(self,state,alpha,beta,d):
#   Pseudocode
        # function alphabeta(node, depth, α, β, maximizingPlayer) is
#     if depth = 0 or node is a terminal node then
#         return the heuristic value of node
#     if maximizingPlayer then
#         v := -∞
#         for each child of node do
#             v := max(v, alphabeta(child, depth – 1, α, β, FALSE))
#             α := max(α, v)
#             if β ≤ α then
#                 break (* β cut-off *)
#         return v
#     else
#         v := +∞
#         for each child of node do
#             v := min(v, alphabeta(child, depth – 1, α, β, TRUE))
#             β := min(β, v)
#             if β ≤ α then
#                 break (* α cut-off *)
#         return v
        def min_value(state, d, alpha, beta):
            if state.terminal_test(): return state.utility(self.player_id)
            if d <= 0: return self.score(state)
            value = float("inf")
            for action in state.actions():
                value = min(value, max_value(state.result(action), d - 1, alpha, beta))
                beta = min(beta, value)
                if beta <= alpha: break
            return value

        def max_value(state, d, alpha, beta):
            if state.terminal_test(): return state.utility(self.player_id)
            if d <= 0: return self.score(state)
            value = float("-inf")
            for action in state.actions():
                value = max(value, min_value(state.result(action), d - 1, alpha, beta))
                alpha = max(alpha, value)
                if beta <= alpha: break
            return value
 
        return max(state.actions(), key=lambda x: min_value(state.result(x), d - 1, alpha,beta))#float("-inf"), float("inf")))


    def mtdf(self,gameState,f, d):
#   Pseudocode      
#    function MTDF(root, f, d) is
#    g := f
#    upperBound := +∞
#    lowerBound := -∞
#    while lowerBound < upperBound do
#        β := max(g, lowerBound+1)
#        g := AlphaBetaWithMemory(root, β-1, β, d)
#        if g < β then
#            upperBound := g 
#        else
#            lowerBound := g
#    return g
        g = f
        upperBound = +1000
        lowerBound = -1000
        while lowerBound < upperBound:
          beta = max(g,lowerBound+1)
          g = CustomPlayer.AlphaBetaWithMemory(self,gameState,beta-1,beta,d)
          if g < beta :
              upperBound = g
          else:
              lowerBound = g
        return g
        

        
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller is responsible for
        cutting off the function after the search time limit has expired. 

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        import random
        f = random.choice(state.actions())
#        best_move = CustomPlayer.iterative(self,state,f,3)
        best_move = CustomPlayer.iterative(self,state,f,3)
        self.queue.put(best_move)






#        import random
#        self.queue.put(random.choice(state.actions()))
