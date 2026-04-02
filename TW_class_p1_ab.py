#%%
import picos as pc
import numpy as np
from itertools import product, combinations
import scipy.sparse as sp
#%%
class TW_ab ():
    def __init__(self, ma=2, mb=2,  kx=3, ky=3,  solver = 'mosek'):
        self.ma = ma
        self.mb = mb
        self.kx = kx
        self.ky = ky
        self.outputs = list(product(range(ma), range(mb)))
        self.inputs = list(product(range(kx), range(ky)))
        self.outputs_twin = list(product(range(ma),range(ma), range(mb), range(mb)))
        self.inputs_twin = list(product(range(kx), range(kx),range(ky), range(ky)))
        self.solver = solver
    
    def pos(self, a, b, x, y):
        return (a*(self.mb) + b , x*(self.ky) + y )

    def canonicalize(self, a, at, b, bt,  x, xt, y, yt):
        if (a, x) > (at, xt):
            a, at = at, a
            x, xt = xt, x
        if (b, y) > (bt, yt):
            b, bt = bt, b
            y, yt = yt, y

        return (a, at, b, bt,  x, xt, y, yt)
    
    def build_index_map(self):
        index_map = {}
        canon_map = {}
        counter = 0

        for (a, at, b, bt) in self.outputs_twin:
            for (x, xt, y, yt) in self.inputs_twin:

                key = (a, at, b, bt, x, xt, y, yt)
                canon = self.canonicalize(*key)

                if canon not in canon_map:
                    canon_map[canon] = counter
                    counter += 1

                index_map[key] = canon_map[canon]

        return index_map, counter
    
    def get_P_twin(self, P_twin, index_map,
               a, at, b, bt,
               x, xt, y, yt):
        return P_twin[index_map[(a, at, b, bt, x, xt, y, yt)]]



    def add_ns_constraints(self, P_twin, problem, index_map):
        # -------- No-signalling A --------
        for x in range(self.kx):
            for a in range(self.ma):
                ref_terms = [
                    self.get_P_twin(P_twin, index_map, a, a_twin, b, b_twin, x, 0, 0, 0)
                    for a_twin, b, b_twin in product(range(self.ma), range(self.mb), range(self.mb))
                ]
                ref = pc.sum(ref_terms)
                for xt,y, yt in product(range(self.kx),range(self.ky), range(self.ky)):
                    terms = []
                    for a_twin,b, b_twin in product(range(self.ma),range(self.mb), range(self.mb) ):
                        val = self.get_P_twin(
                            P_twin, index_map,
                            a, a_twin, b, b_twin,
                            x, xt, y, yt
                        )
                        terms.append(val)
                    expr = pc.sum(terms)
                    problem.add_constraint(expr == ref)

        # -------- No-signalling A' --------
        for a_twin in range(self.ma):
            for xt in range(self.kx):
                ref_terms = [
                    self.get_P_twin(P_twin, index_map, a, a_twin, b, b_twin, 0, xt, 0, 0)
                    for a, b, b_twin in product(range(self.ma), range(self.mb), range(self.mb))
                ]
                ref = pc.sum(ref_terms)
                for x,y, yt, in product(range(self.kx),range(self.ky), range(self.ky)):
                    terms = []
                    for a,b, b_twin in product(range(self.ma),range(self.mb), range(self.mb) ):
                        val = self.get_P_twin(
                            P_twin, index_map,
                            a, a_twin, b, b_twin,
                            x, xt, y, yt
                        )
                        terms.append(val)
                    expr = pc.sum(terms)
                    problem.add_constraint(expr == ref)


        # -------- No-signalling B --------
        for y in range(self.ky):
            for b in range(self.mb):
                ref_terms = [
                    self.get_P_twin(P_twin, index_map, a, a_twin, b, b_twin, 0, 0, y, 0)
                    for a,a_twin,  b_twin in product(range(self.ma), range(self.ma), range(self.mb))
                ]
                ref = pc.sum(ref_terms)
                for x, xt,yt in product(range(self.kx), range(self.kx), range(self.ky)):
                    terms = []
                    for a, a_twin, b_twin in product(range(self.ma), range(self.ma),range(self.mb) ):
                        val = self.get_P_twin(
                            P_twin, index_map,
                            a, a_twin, b, b_twin,
                            x, xt, y, yt
                        )
                        terms.append(val)
                    expr = pc.sum(terms)
                    problem.add_constraint(expr == ref)
        
        # -------- No-signalling B' --------
        for yt in range(self.ky):
            for b_twin in range(self.mb):
                ref_terms = [
                    self.get_P_twin(P_twin, index_map, a, a_twin, b, b_twin, 0, 0, 0, yt)
                    for a,a_twin, b in product(range(self.ma), range(self.ma), range(self.mb))
                ]
                ref = pc.sum(ref_terms)
                for x, xt,y, in product(range(self.kx), range(self.kx), range(self.ky)):
                    terms = []
                    for a, a_twin, b in product(range(self.ma), range(self.ma),range(self.mb)):
                        val = self.get_P_twin(
                            P_twin, index_map,
                            a, a_twin, b, b_twin,
                            x, xt, y, yt, 
                        )
                        terms.append(val)
                    expr = pc.sum(terms)
                    problem.add_constraint(expr == ref)

    def add_ns_constraints_P(self, P, problem):
        # -------- No-signalling A --------
        for x in range(self.kx):
            for a in range(self.ma):
                for y1 in range(self.ky):
                    ref = sum(P[self.pos(a, b, x, y1)] for b in range(self.mb))
                    for y2 in range(self.ky):
                        expr = sum(P[self.pos(a, b, x, y2)] for b in range(self.mb))
                        problem.add_constraint(expr == ref)

        # -------- No-signalling B --------
        for y in range(self.ky):
            for b in range(self.mb):
                for x1 in range(self.kx):
                    ref = sum(P[self.pos(a, b, x1, y)] for a in range(self.ma))
                    for x2 in range(self.kx):
                        expr = sum(P[self.pos(a, b, x2, y)] for a in range(self.ma))
                        problem.add_constraint(expr == ref)

       
    def add_ns_constraints2(self, P_twin, problem, index_map):
        # -------- No-signalling A --------
        # Para cada x, a
        for x in range(self.kx):
            for a in range(self.ma):
                # Referencia: sum sobre a_twin, b, b_twin con xt=0, y=0, yt=0
                ref_terms = []
                for a_twin, b, b_twin in product(range(self.ma), range(self.mb), range(self.mb)):
                    val = self.get_P_twin(
                        P_twin, index_map,
                        a, a_twin, b, b_twin,
                        x, 0, 0, 0
                    )
                    ref_terms.append(val)
                ref = pc.sum(ref_terms)

                # Para cada xt, y, yt
                for xt, y, yt in product(range(self.kx), range(self.ky), range(self.ky)):
                    terms = []
                    for a_twin, b, b_twin in product(range(self.ma), range(self.mb), range(self.mb)):
                        val = self.get_P_twin(
                            P_twin, index_map,
                            a, a_twin, b, b_twin,
                            x, xt, y, yt
                        )
                        terms.append(val)
                    expr = pc.sum(terms)
                    problem.add_constraint(expr == ref)

        # -------- No-signalling A' --------
        for a_twin in range(self.ma):
            for xt in range(self.kx):
                # Referencia: sum sobre a, b, b_twin con x=0, y=0, yt=0
                ref_terms = []
                for a, b, b_twin in product(range(self.ma), range(self.mb), range(self.mb)):
                    val = self.get_P_twin(
                        P_twin, index_map,
                        a, a_twin, b, b_twin,
                        0, xt, 0, 0
                    )
                    ref_terms.append(val)
                ref = pc.sum(ref_terms)

                for x, y, yt in product(range(self.kx), range(self.ky), range(self.ky)):
                    terms = []
                    for a, b, b_twin in product(range(self.ma), range(self.mb), range(self.mb)):
                        val = self.get_P_twin(
                            P_twin, index_map,
                            a, a_twin, b, b_twin,
                            x, xt, y, yt
                        )
                        terms.append(val)
                    expr = pc.sum(terms)
                    problem.add_constraint(expr == ref)

        # -------- No-signalling B --------
        for y in range(self.ky):
            for b in range(self.mb):
                # Referencia: sum sobre a, a_twin, b_twin con x=0, xt=0, yt=0
                ref_terms = []
                for a, a_twin, b_twin in product(range(self.ma), range(self.ma), range(self.mb)):
                    val = self.get_P_twin(
                        P_twin, index_map,
                        a, a_twin, b, b_twin,
                        0, 0, y, 0
                    )
                    ref_terms.append(val)
                ref = pc.sum(ref_terms)

                for x, xt, yt in product(range(self.kx), range(self.kx), range(self.ky)):
                    terms = []
                    for a, a_twin, b_twin in product(range(self.ma), range(self.ma), range(self.mb)):
                        val = self.get_P_twin(
                            P_twin, index_map,
                            a, a_twin, b, b_twin,
                            x, xt, y, yt
                        )
                        terms.append(val)
                    expr = pc.sum(terms)
                    problem.add_constraint(expr == ref)

        # -------- No-signalling B' --------
        for yt in range(self.ky):
            for b_twin in range(self.mb):
                # Referencia: sum sobre a, a_twin, b con x=0, xt=0, y=0
                ref_terms = []
                for a, a_twin, b in product(range(self.ma), range(self.ma), range(self.mb)):
                    val = self.get_P_twin(
                        P_twin, index_map,
                        a, a_twin, b, b_twin,
                        0, 0, 0, yt
                    )
                    ref_terms.append(val)
                ref = pc.sum(ref_terms)

                for x, xt, y in product(range(self.kx), range(self.kx), range(self.ky)):
                    terms = []
                    for a, a_twin, b in product(range(self.ma), range(self.ma), range(self.mb)):
                        val = self.get_P_twin(
                            P_twin, index_map,
                            a, a_twin, b, b_twin,
                            x, xt, y, yt
                        )
                        terms.append(val)
                    expr = pc.sum(terms)
                    problem.add_constraint(expr == ref)

    def normalization_twin(self, P_twin, problem, index_map):
        for x, x_twin, y, y_twin in self.inputs_twin:

            terms = []

            for a, a_twin, b, b_twin in self.outputs_twin:
                val = self.get_P_twin(
                    P_twin, index_map,
                    a, a_twin, b, b_twin,
                    x, x_twin, y, y_twin
                )
                terms.append(val)

            problem.add_constraint(pc.sum(terms) == 1)

    def normalization_P(self, P, problem):
        for x, y in self.inputs:

            terms = []

            for a, b in self.outputs:
                val = P[self.pos(a, b, x, y)]
                terms.append(val)

            problem.add_constraint(pc.sum(terms) == 1)

        
    def relate_P_twin_P(self, P_twin, P, problem, index_map):
        for x, x_twin, y, y_twin in self.inputs_twin:
            for a, b in self.outputs:

                terms = []

                for a_twin, b_twin in self.outputs:
                    val = self.get_P_twin(
                        P_twin, index_map,
                        a, a_twin, b, b_twin,
                        x, x_twin, y, y_twin
                    )
                    terms.append(val)

                problem.add_constraint(
                    P[self.pos(a, b, x, y)] == pc.sum(terms)
                )
        
    def correlacion(self,x, y, P):
        return sum(((-1)**(a+b)) * P[self.pos(a, b, x, y)] for a in range(self.ma) for b in range(self.mb) )        
    
    def E(self,x, y, P):
        return sum(((-1)**(a+b)) * P[self.pos(a, b, x, y)] for a in range(self.ma) for b in range(self.mb) )        
    

    def marginal_A(self, x, P):
        return sum(
        ((-1)**a) * sum(P[self.pos(a,b,x,y)] for b in range(self.mb)) 
            for a in range(self.ma) 
            for y in range(self.ky)
        ) / self.ky  # promedio sobre y

    def marginal_B(self, y, P):
        return sum(
            ((-1)**b) * sum(P[self.pos(a,b,x,y)] for a in range(self.ma)) 
            for b in range(self.mb) 
            for x in range(self.kx)
        ) / self.kx  # promedio sobre x

    def solve_CHSH(self):
        if (self.kx ==2 and self.ky == 2 and self.ma ==2 and self.mb ==2):
            problem = pc.Problem (verbosity =1)
            P = pc.RealVariable("P", (self.ma*self.mb, self.kx*self.ky), lower=0, upper=1)
            index_map, n_vars = self.build_index_map()
            P_twin = pc.RealVariable("P_twin_reduced", n_vars, lower=0, upper=1)
            self.normalization_twin(P_twin, problem, index_map)
            self.add_ns_constraints( P_twin, problem, index_map)
            self.relate_P_twin_P(P_twin, P, problem, index_map)
            self.add_ns_constraints_P( P, problem)
            self.normalization_P(P, problem)




            CHSH = self.correlacion(0,0, P) + self.correlacion(0,1, P) + self.correlacion(1,0, P) - self.correlacion(1,1, P )

            problem.set_objective('max', CHSH)
            problem.solve(solver=self.solver)  # SCS/ECOS también funcionan
            print("Valor máximo de CHSH:", CHSH.value)
        else:
            print("CHSH only defined for ma=mb=2 and kx=ky=2")

    def PA0(self, x, P):
    # P(a=0 | x)
            return sum(
                sum(P[self.pos(0,b,x,y)] for b in range(self.mb))
                for y in range(self.ky)
            ) / self.ky
    
    def PA1(self, x, P):
    # P(a=1 | x)
            return sum(
                sum(P[self.pos(1,b,x,y)] for b in range(self.mb))
                for y in range(self.ky)
            ) / self.ky



    def PB0(self, y, P):
        # P(b=0 | y)
        return sum(
            sum(P[self.pos(a,0,x,y)] for a in range(self.ma))
            for x in range(self.kx)
        ) / self.kx
    
    def PB1(self, y, P):
        # P(b=1 | y)
        return sum(
            sum(P[self.pos(a,1,x,y)] for a in range(self.ma))
            for x in range(self.kx)
        ) / self.kx


    def P00(self, x, y, P):
        return P[self.pos(0,0,x,y)]
    
    def solve_I3322(self):
        if (self.kx ==3 and self.ky == 3 and self.ma ==2 and self.mb ==2):
            problem = pc.Problem (verbosity =1)
            P = pc.RealVariable("P", (self.ma*self.mb, self.kx*self.ky), lower=0, upper=1)
            index_map, n_vars = self.build_index_map()
            P_twin = pc.RealVariable("P_twin_reduced", n_vars, lower=0, upper=1)
            self.add_ns_constraints( P_twin, problem, index_map)
            self.normalization_twin(P_twin, problem, index_map)
            self.relate_P_twin_P(P_twin, P, problem,index_map)
            self.add_ns_constraints_P( P, problem)
            self.normalization_P(P, problem)

            I3322= (
                # ---- correlaciones conjuntas ----
                + self.P00(0,0,P) + self.P00(0,1,P) + self.P00(0,2,P)
                + self.P00(1,0,P) + self.P00(1,1,P) - self.P00(1,2,P)
                + self.P00(2,0,P) - self.P00(2,1,P)

                # ---- marginales ----
                - self.PA0(0,P) 
                - 2*self.PB0(0,P) - self.PB0(1,P)
            )

            problem.set_objective('max', I3322)
            problem.solve(solver=self.solver) 
            print("Valor máximo de I3322:", I3322.value)
        else:
            print("I3322 only defined for ma=mb=2 and kx=ky=3")

    def solve_I3322_vers2(self):
        if (self.kx ==3 and self.ky == 3 and self.ma ==2 and self.mb ==2):
            problem = pc.Problem (verbosity =1)
            P = pc.RealVariable("P", (self.ma*self.mb, self.kx*self.ky), lower=0, upper=1)
            index_map, n_vars = self.build_index_map()
            P_twin = pc.RealVariable("P_twin_reduced", n_vars, lower=0, upper=1)
            self.add_ns_constraints( P_twin, problem, index_map)
            self.normalization_twin(P_twin, problem, index_map)
            self.relate_P_twin_P(P_twin, P, problem, index_map)


            I3322 = (
            self.marginal_A(0,P) + self.marginal_A(1,P) - self.marginal_B(0,P) - self.marginal_B(1,P)
            + self.E(0,0,P) +self.E(0,1,P) + self.E(1,0,P) + self.E(1,1,P) + self.E(2,0,P)
            - self.E(2,1,P) + self.E(0,2,P) - self.E(1,2,P)
        )

            problem.set_objective('max', I3322)
            problem.solve(solver=self.solver) 
            print("Maximum value of I3322:", I3322.value)
        else:
            print("I3322 only defined for ma=mb=2 and kx=ky=3")

    def solve_I4422(self):
        if (self.kx ==4 and self.ky == 4 and self.ma ==2 and self.mb ==2):
            problem = pc.Problem (verbosity =1)
            P = pc.RealVariable("P", (self.ma*self.mb, self.kx*self.ky), lower=0, upper=1)
            index_map, n_vars = self.build_index_map()
            P_twin = pc.RealVariable("P_twin_reduced", n_vars, lower=0, upper=1)
            self.add_ns_constraints( P_twin, problem, index_map)
            self.add_ns_constraints_P( P, problem)
            self.normalization_P(P, problem)
            self.normalization_twin(P_twin, problem, index_map)
            self.relate_P_twin_P(P_twin, P, problem, index_map)
            I4422= (
                # ---- correlaciones conjuntas ----
                + self.P00(0,0,P) + self.P00(0,1,P) + self.P00(0,2,P) + self.P00(0,3,P)
                + self.P00(1,0,P) + self.P00(1,1,P) + self.P00(1,2,P)- self.P00(1,3,P)
                + self.P00(2,0,P) + self.P00(2,1,P) - self.P00(2,2,P)
                + self.P00(3,0,P) - self.P00(3,1,P)

                # ---- marginales ----
                - self.PA0(0,P) 
                - 3*self.PB0(0,P) - 2*self.PB0(1,P) - self.PB0(2,P)
            )

            problem.set_objective('max', I4422)
            problem.solve(solver=self.solver) 
            print("Valor máximo de I4422:", I4422.value)
        else:
            print("I4422 only defined for ma=mb=2 and kx=ky=4")


    def solve_I2233(self):
        if (self.kx ==2 and self.ky == 2 and self.ma ==3 and self.mb ==3):
            problem = pc.Problem (verbosity =1)
            P = pc.RealVariable("P", (self.ma*self.mb, self.kx*self.ky), lower=0, upper=1)
            index_map, n_vars = self.build_index_map()
            P_twin = pc.RealVariable("P_twin_reduced", n_vars, lower=0, upper=1)
            self.add_ns_constraints( P_twin, problem, index_map)
            self.normalization_twin(P_twin, problem, index_map)
            self.relate_P_twin_P(P_twin, P, problem,index_map)
            self.add_ns_constraints_P( P, problem)
            self.normalization_P(P, problem)

            I2233= (
                # ---- correlaciones conjuntas ----
                P[self.pos(0,0,0,0)] + P[self.pos(0,1,0,0)] +  P[self.pos(0,1,0,1)]
                + P[self.pos(1,0,0,0)] +  P[self.pos(1,0,0,1)] +  P[self.pos(1,1,0,1)]
                +  P[self.pos(0,1,1,0)] -  P[self.pos(0,1,1,1)]
                +  P[self.pos(1,0,1,0)] +  P[self.pos(1,1,1,0)]  -  P[self.pos(1,0,1,1)] -  P[self.pos(1,1,1,1)]

                # ---- marginales ----
                - self.PA0(0,P)  - self.PA1(0,P) 
                - self.PB0(0,P) - self.PB1(0,P)
            )

            problem.set_objective('max', I2233)
            problem.solve(solver=self.solver) 
            print("Valor máximo de I2233:", I2233.value)
        else:
            print("I2233 only defined for ma=mb=3 and kx=ky=2")
#%%
if __name__ == "__main__":
    tw = TW_ab(ma=3, mb=3,  kx=2, ky=2, solver = 'mosek')
    #tw.solve_CHSH()
    #tw.solve_I3322()
    #tw.solve_I4422()
    tw.solve_I2233()
    
# %%
