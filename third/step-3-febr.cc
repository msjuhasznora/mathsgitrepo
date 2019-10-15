/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2018 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------
 *
 * Authors: Wolfgang Bangerth, 1999,
 *          Guido Kanschat, 2011
 */
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/grid/grid_refinement.h>
#include <math.h>
#include <fstream>
#include <iostream>

using namespace dealii;

class Step3
{
public:
    Step3 ();
    void run ();
private:
    
    struct ScratchData{
        std::vector<double> rhs_values;
        FEValues<2> fe_values;
        
        ScratchData(const FiniteElement<2> &fe, const Quadrature<2> &quadrature, const UpdateFlags update_flags):
        rhs_values(quadrature.size()),
        fe_values(fe, quadrature, update_flags){}
        
        ScratchData(const ScratchData &rhs):
            rhs_values(rhs.rhs_values),
            fe_values(rhs.fe_values.get_fe(),
                      rhs.fe_values.get_quadrature(),
                      rhs.fe_values.get_update_flags()
                
        )
        {}
    };
    
    struct PerTaskData{
        
        FullMatrix<double> cell_matrix;
        Vector<double> cell_rhs;
        std::vector<unsigned int> dof_indices;
        
        PerTaskData(const FiniteElementData<2> &fe):
            cell_matrix(fe.dofs_per_cell,
                        fe.dofs_per_cell),
            cell_rhs (fe.dofs_per_cell),
            dof_indices (fe.dofs_per_cell)
        {}
    };
    
    void make_grid (int refine);
    void assemble_system_one_cell(DoFHandler<2>::active_cell_iterator cell,
                                  ScratchData &scratch, PerTaskData &copy);
    void local_to_global(PerTaskData &copy);
    void setup_system ();
    void kelly_calc ();
    void refine_grid ();
    void assemble_system ();
    void solve ();
    void output_results () const;
    double error_calc ();
    Triangulation<2>        triangulation;
    FE_Q<2>                 fe;
    DoFHandler<2>           dof_handler;
    ConstraintMatrix        constraints;
    SparsityPattern         sparsity_pattern;
    SparseMatrix<double>    system_matrix;
    mutable Vector<double>  solution;
    Vector<double>          system_rhs;
};

Step3::Step3 ()
:
fe (1),
dof_handler (triangulation)
{}

void Step3::make_grid (int refine)
{
    GridGenerator::hyper_cube (triangulation, -1, 1);
    triangulation.refine_global (refine);
    std::cout << "Number of active cells: "
    << triangulation.n_active_cells()
    << std::endl;
}

void Step3::setup_system ()
{
    dof_handler.distribute_dofs (fe);
    std::cout << "Number of degrees of freedom: "
    << dof_handler.n_dofs()
    << std::endl;
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern (dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit (sparsity_pattern);
    solution.reinit (dof_handler.n_dofs());
    system_rhs.reinit (dof_handler.n_dofs());
    
    constraints.clear ();
    DoFTools::make_hanging_node_constraints (dof_handler,
                                             constraints);
    VectorTools::interpolate_boundary_values (dof_handler,
                                              0,
                                              Functions::ZeroFunction<2>(),
                                              constraints);
    constraints.close ();
}

void Step3::assemble_system_one_cell (DoFHandler<2>::active_cell_iterator cell,
                              ScratchData &scratch, PerTaskData &data){
    scratch.fe_values.reinit (cell);
    data.cell_matrix = 0;
    data.cell_rhs = 0;
    
    for (unsigned int q_index=0; q_index<scratch.fe_values.get_quadrature().size(); ++q_index)
    {
        for (unsigned int i=0; i<scratch.fe_values.get_fe().dofs_per_cell; ++i)
            for (unsigned int j=0; j<scratch.fe_values.get_fe().dofs_per_cell; ++j)
                data.cell_matrix(i,j) += (scratch.fe_values.shape_grad (i, q_index) *
                                     scratch.fe_values.shape_grad (j, q_index) *
                                     scratch.fe_values.JxW (q_index));
        for (unsigned int i=0; i<scratch.fe_values.get_fe().dofs_per_cell; ++i)
            data.cell_rhs(i) += (scratch.fe_values.shape_value (i, q_index) *
                            40 * (numbers::PI)* (numbers::PI)* sin(2*(numbers::PI)*scratch.fe_values.quadrature_point(q_index)[0])*
                            sin(6*(numbers::PI)*scratch.fe_values.quadrature_point(q_index)[1])*
                            scratch.fe_values.JxW (q_index));
    }
    cell->get_dof_indices (data.dof_indices);
}

void Step3::local_to_global (PerTaskData &data){
    
    constraints.distribute_local_to_global (data.cell_matrix,
                                            data.cell_rhs,
                                            data.dof_indices,
                                            system_matrix,
                                            system_rhs);
}

void Step3::assemble_system ()
{
    QGauss<2> quadrature_formula(2);
    FEValues<2> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients | update_JxW_values | update_quadrature_points);
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points    = quadrature_formula.size();
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    
    ScratchData scratch(fe, quadrature_formula, update_values | update_gradients | update_JxW_values | update_quadrature_points);
    
    PerTaskData data (fe);
    
    for (const auto &cell: dof_handler.active_cell_iterators())
    {
        assemble_system_one_cell(cell, scratch, data);
        /*
        scratch.fe_values.reinit (cell);
        data.cell_matrix = 0;
        data.cell_rhs = 0;
        for (unsigned int q_index=0; q_index<scratch.fe_values.get_quadrature().size(); ++q_index)
        {
            for (unsigned int i=0; i<scratch.fe_values.get_fe().dofs_per_cell; ++i)
                for (unsigned int j=0; j<scratch.fe_values.get_fe().dofs_per_cell; ++j)
                    data.cell_matrix(i,j) += (scratch.fe_values.shape_grad (i, q_index) *
                                         scratch.fe_values.shape_grad (j, q_index) *
                                         scratch.fe_values.JxW (q_index));
            for (unsigned int i=0; i<scratch.fe_values.get_fe().dofs_per_cell; ++i)
                data.cell_rhs(i) += (scratch.fe_values.shape_value (i, q_index) *
                                40 * (numbers::PI)* (numbers::PI)* sin(2*(numbers::PI)*scratch.fe_values.quadrature_point(q_index)[0])*
                            sin(6*(numbers::PI)*scratch.fe_values.quadrature_point(q_index)[1])*
                                scratch.fe_values.JxW (q_index));
        } */
        
        local_to_global(data);
        
        /* std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
        
        cell->get_dof_indices (local_dof_indices);
        constraints.distribute_local_to_global (data.cell_matrix,
                                                data.cell_rhs,
                                                data.dof_indices,
                                                system_matrix,
                                                system_rhs); */
        
    }
    
    std::map<types::global_dof_index,double> boundary_values;
    VectorTools::interpolate_boundary_values (dof_handler,
                                              0,
                                              Functions::ZeroFunction<2>(),
                                              boundary_values);
    MatrixTools::apply_boundary_values (boundary_values,
                                        system_matrix,
                                        solution,
                                        system_rhs);
}

void Step3::solve ()
{
    SolverControl           solver_control (1000, 1e-12);
    SolverCG<>              solver (solver_control);
    solver.solve (system_matrix, solution, system_rhs,
                  PreconditionIdentity());
    constraints.distribute (solution);
}

void Step3::kelly_calc ()
{
    Vector<float> estimated_error_per_cell (triangulation.n_active_cells());
    KellyErrorEstimator<2>::estimate (dof_handler,
                                        QGauss<1>(2),
                                        typename FunctionMap<2>::type(),
                                        solution,
                                        estimated_error_per_cell);
    std::cout << "*** Kelly Error Estimator ***" << std::endl;
    // DataOut::add_data_vector
    std::string vtk_filename = "estimated_error_per_cell.vtk";
    std::ofstream output2(vtk_filename);
    DataOut<2> data_out2;
    data_out2.attach_dof_handler(dof_handler);
    data_out2.add_data_vector(estimated_error_per_cell, "estimated_error_per_cell");
    data_out2.build_patches();
    data_out2.write_vtk(output2);

}

void Step3::refine_grid ()
{
    Vector<float> estimated_error_per_cell (triangulation.n_active_cells());
    KellyErrorEstimator<2>::estimate (dof_handler,
                                        QGauss<2-1>(2),
                                        typename FunctionMap<2>::type(),
                                        solution,
                                        estimated_error_per_cell);
    GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                                     estimated_error_per_cell,
                                                     0.3, 0.03);
    triangulation.execute_coarsening_and_refinement ();
}

void Step3::output_results () const
{
    DataOut<2> data_out;
    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (solution, "solution");
    data_out.build_patches ();
    std::ofstream output ("solution.vtk");
    data_out.write_vtk (output);
}

double Step3::error_calc ()
{
    std::map<std::string,double> constants;
    constants["pi"] = numbers::PI;
    
    std::string variables = "x,y";
    std::vector<std::string> expressions(1);
    expressions[0] = "sin(2*pi*x)*sin(6*pi*y)";
    
    FunctionParser<2> vector_function(1);
    vector_function.initialize(variables, expressions, constants);
    
    Vector<double> interpolatedSolution;
    interpolatedSolution.reinit (dof_handler.n_dofs());
    
    VectorTools::interpolate (dof_handler, vector_function, interpolatedSolution);
    
    // calculate the basic l_infty norm for the difference of 2 vectors
    Vector<double> difference(solution);
    difference -= interpolatedSolution;
    double error = difference.linfty_norm();
    std::cout << "L-infty vector error: " << error << std::endl;
    
    // calculate with VectorTools::IntegrateDifference
    
    Vector<float> difference_per_cell(triangulation.n_active_cells());
    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      vector_function,
                                      difference_per_cell,
                                      QGauss<2>(2),
                                      VectorTools::Linfty_norm);
    const double Linfty_error =
    VectorTools::compute_global_error(triangulation,
                                      difference_per_cell,
                                      VectorTools::Linfty_norm);
    
    std::cout << "L-infty vector error by diff per cell: " << Linfty_error << std::endl;
    
    
    // DataOut::add_data_vector
    std::string vtk_filename = "errorvisualization.vtk";
    std::ofstream output(vtk_filename);
    DataOut<2> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(difference_per_cell, "difference_per_cell");
    data_out.build_patches();
    data_out.write_vtk(output);

    return Linfty_error;
    
}


void Step3::run ()
{
    make_grid (5);
    setup_system ();
    assemble_system ();
    solve ();
    output_results ();
    error_calc();
    kelly_calc();
}

int main ()
{
    deallog.depth_console (2);
    Step3 laplace_problem;
    laplace_problem.run ();
    return 0;
}
