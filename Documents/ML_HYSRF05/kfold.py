# kfold.py — 1D linear regression with K-fold CV (no external deps)
# Usage:
#   from kfold import kfold_linear_1d
#   result = kfold_linear_1d(x, y, k=5, shuffle=True, seed=42, standardize_per_fold=False)

from typing import Dict, List, Tuple, Any
import numpy as np
import math, copy
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
from render_figure import plt_house_x, plt_contour_wgrad, plt_divergence, plt_gradients

def compute_cost(x, y, w, b): 
    """
    Computes the cost function for linear regression.
    
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # number of training examples
    m = x.shape[0] 
    
    cost_sum = 0 
    for i in range(m): 
        f_wb = w * x[i] + b   
        cost = (f_wb - y[i]) ** 2  
        cost_sum = cost_sum + cost  
    total_cost = (1 / (2 * m)) * cost_sum  

    return total_cost

def compute_gradient(x, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """
    
    # Number of training examples
    m = x.shape[0]    
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):  
        f_wb = w * x[i] + b 
        dj_dw_i = (f_wb - y[i]) * x[i] 
        dj_db_i = f_wb - y[i] 
        dj_db += dj_db_i
        dj_dw += dj_dw_i 
    dj_dw = dj_dw / m 
    dj_db = dj_db / m 
        
    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function): 
    """
    Performs gradient descent to fit w,b. Updates w,b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      x (ndarray (m,))  : Data, m examples 
      y (ndarray (m,))  : target values
      w_in,b_in (scalar): initial values of model parameters  
      alpha (float):     Learning rate
      num_iters (int):   number of iterations to run gradient descent
      cost_function:     function to call to produce cost
      gradient_function: function to call to produce gradient
      
    Returns:
      w (scalar): Updated value of parameter after running gradient descent
      b (scalar): Updated value of parameter after running gradient descent
      J_history (List): History of cost values
      p_history (list): History of parameters [w,b] 
      """
    
    w = copy.deepcopy(w_in) # avoid modifying global w_in
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    p_history = []
    b = b_in
    w = w_in
    
    for i in range(num_iters):
        # Calculate the gradient and update the parameters using gradient_function
        dj_dw, dj_db = gradient_function(x, y, w , b)     

        # Update Parameters using equation (3) above
        b = b - alpha * dj_db                            
        w = w - alpha * dj_dw                            

        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(x, y, w , b))
            p_history.append([w,b])
        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
 
    return w, b, J_history, p_history #return w and J,w history for graphing

def find_w_and_b(x_s, y_s):
    w_init = 0
    b_init = 0
    # some gradient descent settings
    iterations = 10000
    tmp_alpha = 1.0e-2
    # run gradient descent
    mx = x_s.mean();
    sx = x_s.std(ddof=0);
    my = y_s.mean();
    sy = y_s.std(ddof=0);
    w_final, b_final, J_hist, p_hist = gradient_descent(x_s ,y_s, w_init, b_init, tmp_alpha, 
                                                        iterations, compute_cost, compute_gradient)
    print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")
    def convert_to_original_params(w_s, b_s, mx, sx, my):
        w = w_s / sx
        b = my + b_s - w * mx
        return w, b
    w, b = convert_to_original_params(w_final, b_final, mx, sx, my)
    print(w, b) 


def simulation(x_fold, y_fold, *,
               fig_name=None,
               use_style='./deeplearning.mplstyle',
               block=True,
               save_path=None):
    import numpy as np
    import matplotlib.pyplot as plt
    from render_figure import plt_stationary, plt_update_onclick

    # Chuẩn hoá theo fold (giống code của bạn)
    mx, sx = x_fold.mean(), x_fold.std(ddof=0)
    if sx == 0: sx = 1.0
    my     = y_fold.mean()
    x_s    = (x_fold - mx)/sx
    y_s    = (y_fold - my)

    # Style (nếu có)
    if use_style:
        try:
            plt.style.use(use_style)
        except Exception:
            pass

    # Gọi đúng chữ ký của bạn: KHÔNG truyền fig/ax
    fig, ax, dyn_items = plt_stationary(x_s, y_s)
    updater = plt_update_onclick(fig, ax, x_s, y_s, dyn_items)

    # Gắn tiêu đề nếu có tên fold
    if fig_name:
        try:
            ax.set_title(fig_name)
        except Exception:
            try:
                fig.suptitle(fig_name)
            except Exception:
                pass

    # Lưu snapshot trạng thái ban đầu nếu cần
    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")

    # Hiển thị tương tác
    if block:
        # Jupyter: dùng %matplotlib notebook hoặc %matplotlib widget trước đó
        import matplotlib
        plt.show(block=True)
    else:
        plt.pause(0.001)

    return fig, ax, updater

def kfold_linear_1d(
    x, y, k=5,
    shuffle=True, seed=42,
    standardize_per_fold=False,  # bạn đang chuẩn hoá riêng; thường để False
    verbose=False,
    interactive=True,            # thêm tham số
    save_dir=None,               # nếu muốn lưu ảnh từng fold
):
    import numpy as np
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = len(x)

    if n != len(y): raise ValueError("x và y phải cùng kích thước")
    if k < 2 or k > n: raise ValueError("k phải nằm trong [2, len(x)]")

    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

    import os
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    folds = np.array_split(idx, k)
    results = []

    for i, val_idx in enumerate(folds, start=1):
        train_idx = np.concatenate([folds[j] for j in range(k) if j != (i-1)])
        x_tr, y_tr = x[train_idx], y[train_idx]
        x_val, y_val = x[val_idx], y[val_idx]

        if verbose:
            print(f"\n===== Fold {i} =====")
            print("x_val:", x_val)
            print("y_val:", y_val)

        # Fit tuyến tính y = w*x + b trên TRAIN
        mx, my = x_tr.mean(), y_tr.mean()
        w = np.dot(x_tr - mx, y_tr - my) / np.dot(x_tr - mx, x_tr - mx)
        b = my - w * mx

        # Đánh giá trên VAL
        y_pred = w * x_val + b
        mse = np.mean((y_pred - y_val)**2)
        mae = np.mean(np.abs(y_pred - y_val))
        ss_res = np.sum((y_val - y_pred)**2)
        ss_tot = np.sum((y_val - y_val.mean())**2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0

        results.append({
            "fold": i,
            "w": w, "b": b,
            "MSE": mse, "MAE": mae, "R2": r2,
            "x_val": x_val, "y_val": y_val,
        })

        # --- GỌI FIGURE TƯƠNG TÁC TỪ simulation ---
        if interactive:
            save_path = f"{save_dir}/fold_{i:02d}.png" if save_dir else None
            simulation(x_val, y_val,
                       fig_name=f"Fold {i}",
                       block=True,
                       save_path=save_path)
            
        
        
    avg = {
        "MSE": np.mean([r["MSE"] for r in results]),
        "MAE": np.mean([r["MAE"] for r in results]),
        "R2": np.mean([r["R2"] for r in results]),
    }
    return {"folds": results, "averages": avg}
