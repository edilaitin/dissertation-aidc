from engine.solvers.impl.SMT_Z3.SMT_Solver_Z3_Int_SB_AllCombinationsOffers_ILP \
    import Z3_SolverInt_SB_Enc_AllCombinationsOffers_ILP
from engine.problem.ProblemDefinition import ManeuverProblem


class Engine(object):
    def __init__(self):
        self.solver = Z3_SolverInt_SB_Enc_AllCombinationsOffers_ILP()

    def solve(self, applicationDef, offers, sb_option=None):
        availableConfigurations = []
        for key, value in offers.items():
            specs_list = [key, value["cpu"], value["memory"], value["storage"], value["price"]]
            availableConfigurations.append(specs_list)

        problem = ManeuverProblem()
        problem.readConfigurationJSON(applicationDef, availableConfigurations)

        self.solver.init_problem(problem, 'optimize', sb_option=sb_option)
        min_price, price_vms, t, a, vms_type = self.solver.run()
        print("DONE")
        print(min_price)
        print(price_vms)
        print(t)
        print(a)
        print(vms_type)
        #  TODO FORMAT OUTPUT


