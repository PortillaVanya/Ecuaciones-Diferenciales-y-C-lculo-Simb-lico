import sys
import argparse
from typing import Tuple, Optional, List, Dict
import math
from dataclasses import dataclass
from enum import Enum

# Importaciones numéricas y gráficas
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.optimize import curve_fit

# Importaciones de cálculo simbólico
import sympy as sp
from sympy import symbols, Function, Eq, Derivative, integrate, exp, log, sqrt, sin, cos, tan

# Importaciones de interfaz gráfica
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
from tkinter import scrolledtext

# Configuración de estilos
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    try:
        plt.style.use('seaborn')
    except OSError:
        pass  # Usar estilo por defecto si ninguno está disponible

class ModelType(Enum):
    """Tipos de modelos disponibles"""
    COOLING = "Enfriamiento de Newton"
    RADIOACTIVE = "Decaimiento Radioactivo"
    EXACT_DE = "ED Exactas"
    NONEXACT_DE = "ED No Exactas"
    SYMBOLIC = "Cálculo Simbólico"
    PARTIAL_DERIV = "Derivadas Parciales"

@dataclass
class ModelParams:
    """Parámetros generales para modelos"""
    T0: float = 95.0
    Tenv: float = 25.0
    k: float = 0.1
    tmax: float = 50.0
    noise: float = 0.5
    N0: int = 1000
    half_life: float = 10.0
    sim_count: int = 5

class PhysicsModels:
    """Clase principal para todos los modelos físicos y matemáticos"""
    
    # ===== LEY DE ENFRIAMIENTO DE NEWTON =====
    @staticmethod
    def newton_analytic(T0: float, T_env: float, k: float, t: np.ndarray) -> np.ndarray:
        """Solución analítica de la ley de enfriamiento de Newton"""
        return T_env + (T0 - T_env) * np.exp(-k * t)
    
    @staticmethod
    def newton_euler(T0: float, T_env: float, k: float, t_grid: np.ndarray) -> np.ndarray:
        """Solución numérica usando método de Euler"""
        T = np.empty_like(t_grid)
        T[0] = T0
        for i in range(1, len(t_grid)):
            dt = t_grid[i] - t_grid[i-1]
            T[i] = T[i-1] - k * (T[i-1] - T_env) * dt
        return T
    
    # ===== DECAIMIENTO RADIOACTIVO =====
    @staticmethod
    def radioactive_analytic(N0: int, half_life: float, t: np.ndarray) -> np.ndarray:
        """Solución analítica del decaimiento radioactivo"""
        lam = math.log(2) / half_life
        return N0 * np.exp(-lam * t)
    
    @staticmethod
    def radioactive_stochastic(N0: int, half_life: float, t_grid: np.ndarray, 
                             rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Simulación estocástica Monte Carlo del decaimiento"""
        if rng is None:
            rng = np.random.default_rng()
        
        lam = math.log(2) / half_life
        N = int(N0)
        counts = np.empty_like(t_grid, dtype=int)
        counts[0] = N
        
        for i in range(1, len(t_grid)):
            dt = t_grid[i] - t_grid[i-1]
            p_decay = 1.0 - math.exp(-lam * dt)
            decays = rng.binomial(N, p_decay)
            N -= decays
            counts[i] = N
            
        return counts
    
    # ===== ECUACIONES DIFERENCIALES EXACTAS =====
    @staticmethod
    def exact_de_solution(M_expr: str, N_expr: str, x0: float, y0: float) -> Dict:
        """
        Resuelve una EDO exacta de la forma M(x,y)dx + N(x,y)dy = 0
        
        Retorna:
            dict con solución, pasos y verificación
        """
        x, y = symbols('x y')
        M = sp.sympify(M_expr)
        N = sp.sympify(N_expr)
        
        # Verificar si es exacta: dM/dy == dN/dx
        dM_dy = sp.diff(M, y)
        dN_dx = sp.diff(N, x)
        is_exact = sp.simplify(dM_dy - dN_dx) == 0
        
        steps = []
        steps.append(f"Ecuación: {M} dx + {N} dy = 0")
        steps.append(f"∂M/∂y = {dM_dy}")
        steps.append(f"∂N/∂x = {dN_dx}")
        steps.append(f"¿Es exacta? {'Sí' if is_exact else 'No'}")
        
        if not is_exact:
            return {
                'exact': False,
                'steps': steps,
                'solution': None,
                'verification': None
            }
        
        try:
            # Encontrar F(x,y) tal que dF/dx = M y dF/dy = N
            F_x = integrate(M, x)
            
            # Encontrar g(y) a partir de dF/dy = N
            dF_dy = sp.diff(F_x, y)
            g_prime = N - dF_dy
            g = integrate(g_prime, y)
            
            F = sp.simplify(F_x + g)
            
            # Calcular F en el punto (x0, y0)
            F_at_point = F.subs({x: x0, y: y0})
            
            steps.append(f"F(x,y) = {F}")
            steps.append(f"F({x0},{y0}) = {F_at_point}")
            
            # Verificar si F_at_point es numérico o constante
            if F_at_point.is_constant():
                # Si es constante, podemos encontrar la solución particular
                constant_value = sp.nsimplify(F_at_point)
                solution_eq = Eq(F, constant_value)
                steps.append(f"Solución particular: {solution_eq}")
            else:
                # Mostrar solución general
                C = sp.symbols('C')
                solution_eq = Eq(F, C)
                steps.append(f"Solución general: {solution_eq}")
                steps.append(f"Nota: Para solución particular, resuelve C = F({x0},{y0}) = {F_at_point}")
            
            # Verificación
            verification_x = sp.simplify(sp.diff(F, x) - M)
            verification_y = sp.simplify(sp.diff(F, y) - N)
            
            steps.append(f"Verificación dF/dx - M = {verification_x}")
            steps.append(f"Verificación dF/dy - N = {verification_y}")
            
            return {
                'exact': True,
                'steps': steps,
                'solution': str(solution_eq),
                'function_F': str(F),
                'F_at_point': str(F_at_point),
                'verification_x': str(verification_x),
                'verification_y': str(verification_y),
                'is_constant': F_at_point.is_constant()
            }
            
        except Exception as e:
            steps.append(f"Error durante la solución: {str(e)}")
            return {
                'exact': True,
                'steps': steps,
                'solution': f"Error: {str(e)}",
                'function_F': "No se pudo determinar",
                'verification': "No se pudo verificar"
            }
    
    # ===== ECUACIONES DIFERENCIALES NO EXACTAS =====
    @staticmethod
    def find_integrating_factor(M_expr: str, N_expr: str) -> Dict:
        """
        Encuentra factor integrante para ecuaciones no exactas
        """
        x, y = symbols('x y')
        M = sp.sympify(M_expr)
        N = sp.sympify(N_expr)
        
        steps = []
        steps.append(f"Ecuación: {M} dx + {N} dy = 0")
        
        # Verificar si ya es exacta
        dM_dy = sp.diff(M, y)
        dN_dx = sp.diff(N, x)
        
        steps.append(f"∂M/∂y = {dM_dy}")
        steps.append(f"∂N/∂x = {dN_dx}")
        
        if sp.simplify(dM_dy - dN_dx) == 0:
            steps.append(f"∂M/∂y = ∂N/∂x → La ecuación YA ES EXACTA")
            return {
                'needs_factor': False,
                'steps': steps,
                'factor': "1"
            }
        
        steps.append(f"∂M/∂y ≠ ∂N/∂x → La ecuación NO es exacta")
        steps.append("")
        
        # Intentar factor integrante μ(x)
        steps.append("Intentando factor integrante μ(x)...")
        expr1 = (dM_dy - dN_dx) / N
        steps.append(f"(∂M/∂y - ∂N/∂x)/N = {expr1}")
        
        try:
            if not expr1.has(y):
                mu_x = sp.exp(integrate(expr1, x))
                steps.append(f"∫({expr1}) dx = {integrate(expr1, x)}")
                steps.append(f"Factor integrante μ(x) = exp(∫) = {mu_x}")
                return {
                    'needs_factor': True,
                    'steps': steps,
                    'factor': str(mu_x),
                    'type': 'μ(x)'
                }
            else:
                steps.append(f"No es función solo de x (contiene y)")
        except Exception as e:
            steps.append(f"Error al calcular μ(x): {str(e)}")
        
        steps.append("")
        
        # Intentar factor integrante μ(y)
        steps.append("Intentando factor integrante μ(y)...")
        expr2 = (dN_dx - dM_dy) / M
        steps.append(f"(∂N/∂x - ∂M/∂y)/M = {expr2}")
        
        try:
            if not expr2.has(x):
                mu_y = sp.exp(integrate(expr2, y))
                steps.append(f"∫({expr2}) dy = {integrate(expr2, y)}")
                steps.append(f"Factor integrante μ(y) = exp(∫) = {mu_y}")
                return {
                    'needs_factor': True,
                    'steps': steps,
                    'factor': str(mu_y),
                    'type': 'μ(y)'
                }
            else:
                steps.append(f"No es función solo de y (contiene x)")
        except Exception as e:
            steps.append(f"Error al calcular μ(y): {str(e)}")
        
        steps.append("")
        steps.append("No se pudo encontrar factor integrante simple")
        steps.append("Se necesitan métodos más avanzados")
        
        return {
            'needs_factor': True,
            'steps': steps,
            'factor': None,
            'type': 'No encontrado'
        }
    
    # ===== CÁLCULO SIMBÓLICO =====
    @staticmethod
    def symbolic_derivative(expr_str: str, var: str, n: int = 1) -> Dict:
        """Calcula derivada simbólica de orden n"""
        x = symbols(var)
        expr = sp.sympify(expr_str)
        
        derivative = sp.diff(expr, x, n)
        
        return {
            'original': str(expr),
            'derivative': str(derivative),
            'latex': sp.latex(derivative),
            'simplified': str(sp.simplify(derivative))
        }
    
    @staticmethod
    def symbolic_integral(expr_str: str, var: str, definite: bool = False, 
                         limits: Tuple[float, float] = None) -> Dict:
        """Calcula integral simbólica"""
        x = symbols(var)
        expr = sp.sympify(expr_str)
        
        try:
            if definite and limits:
                a, b = limits
                integral = integrate(expr, (x, a, b))
                result_type = f"∫_{{{a}}}^{{{b}}} {expr_str} d{var}"
            else:
                integral = integrate(expr, x)
                result_type = f"∫ {expr_str} d{var}"
            
            return {
                'success': True,
                'integral': str(integral),
                'latex': sp.latex(integral),
                'type': result_type,
                'definite': definite
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'integral': f"No se pudo calcular la integral: {str(e)}",
                'type': result_type if 'result_type' in locals() else "Integral"
            }
    
    # ===== DERIVADAS PARCIALES =====
    @staticmethod
    def partial_derivatives(expr_str: str, vars_str: str, order: int = 1) -> Dict:
        """Calcula derivadas parciales de funciones multivariables"""
        vars_list = [symbols(v.strip()) for v in vars_str.split(',')]
        expr = sp.sympify(expr_str)
        
        results = []
        
        if order == 1:
            # Primeras derivadas parciales
            for var in vars_list:
                deriv = sp.diff(expr, var)
                results.append({
                    'variable': str(var),
                    'derivative': str(deriv),
                    'latex': f"\\frac{{\\partial}}{{\\partial {var}}}({expr_str}) = {sp.latex(deriv)}"
                })
        else:
            # Derivadas de orden superior
            for i, var1 in enumerate(vars_list):
                for j, var2 in enumerate(vars_list):
                    if i <= j:  # Evitar duplicados simétricos
                        deriv = sp.diff(expr, var1, var2)
                        results.append({
                            'variables': f"∂²/∂{var1}∂{var2}",
                            'derivative': str(deriv),
                            'latex': f"\\frac{{\\partial^2}}{{\\partial {var1}\\partial {var2}}}({expr_str}) = {sp.latex(deriv)}"
                        })
        
        return {
            'success': True,
            'function': expr_str,
            'variables': vars_str,
            'order': order,
            'results': results
        }

# ===== UTILIDADES =====
def calculate_auto_k(T0: float, T_env: float) -> float:
    """Calcula automáticamente la constante k basada en las temperaturas"""
    delta_T = abs(T0 - T_env)
    
    if delta_T < 0.1:
        delta_T = 0.1
    
    base_k = 0.1
    
    if delta_T < 10:
        scale_factor = 2.0
    elif delta_T > 50:
        scale_factor = 0.3
    else:
        scale_factor = 1.0
    
    auto_k = base_k * scale_factor * (10.0 / delta_T)
    return max(0.005, min(0.5, auto_k))

def calculate_auto_tmax(T0: float, T_env: float, k: float) -> float:
    """Calcula automáticamente el tiempo máximo"""
    if k <= 0:
        k = 0.1
    
    t_95 = -math.log(0.05) / k
    
    if t_95 < 10:
        return math.ceil(t_95)
    elif t_95 < 100:
        return math.ceil(t_95 / 5) * 5
    else:
        return math.ceil(t_95 / 10) * 10

# ===== INTERFAZ GRÁFICA PRINCIPAL =====
class IntegratedMathApp(tk.Tk):
    """Aplicación principal con interfaz gráfica mejorada"""
    
    # Paleta de colores profesional
    COLORS = {
        'bg': '#1a1a2e',
        'sidebar': '#16213e',
        'sidebar_header': '#0f3460',
        'accent': '#e94560',
        'accent_hover': '#ff6b81',
        'btn_normal': '#16213e',
        'btn_hover': '#1a3a5c',
        'btn_active': '#e94560',
        'text_primary': '#eaeaea',
        'text_secondary': '#a0a0b0',
        'content_bg': '#0f0f23',
        'input_bg': '#2a2a4a',
        'input_fg': '#eaeaea',
        'success': '#2ecc71',
        'warning': '#f39c12',
        'error': '#e74c3c',
    }
    
    def __init__(self):
        super().__init__()
        
        self.title("🔬 Modelos Físicos, ED y Cálculo Simbólico")
        self.geometry('1400x850')
        self.minsize(1100, 700)
        
        # Centrar ventana en pantalla
        self.update_idletasks()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        x = (sw - 1400) // 2
        y = (sh - 850) // 2
        self.geometry(f'1400x850+{x}+{y}')
        
        self.params = ModelParams()
        self.current_model = ModelType.COOLING
        
        self._setup_styles()
        self._create_widgets()
        self._create_status_bar()
        self._setup_defaults()
        
    def _setup_styles(self):
        """Configura estilos profesionales para la aplicación"""
        style = ttk.Style()
        try:
            style.theme_use('clam')
        except tk.TclError:
            pass
        
        c = self.COLORS
        self.configure(bg=c['bg'])
        
        # Estilos de ttk personalizados
        style.configure('TFrame', background=c['bg'])
        style.configure('TLabel', background=c['bg'], foreground=c['text_primary'],
                       font=('Segoe UI', 10))
        style.configure('TButton', font=('Segoe UI', 10), padding=6)
        style.configure('TEntry', fieldbackground=c['input_bg'], foreground=c['input_fg'])
        style.configure('TLabelframe', background=c['bg'], foreground=c['text_primary'])
        style.configure('TLabelframe.Label', background=c['bg'], foreground=c['accent'],
                       font=('Segoe UI', 10, 'bold'))
        style.configure('TNotebook', background=c['bg'])
        style.configure('TNotebook.Tab', font=('Segoe UI', 9, 'bold'), padding=[12, 6])
        style.configure('TRadiobutton', background=c['bg'], foreground=c['text_primary'],
                       font=('Segoe UI', 10))
    
    def _create_status_bar(self):
        """Crea barra de estado inferior"""
        c = self.COLORS
        self.status_bar = tk.Label(self, text="✅ Listo — Selecciona un modelo para comenzar",
                                  bg=c['sidebar'], fg=c['text_secondary'],
                                  font=('Segoe UI', 9), anchor='w', padx=10, pady=4)
        self.status_bar.pack(side='bottom', fill='x')
    
    def _update_status(self, message: str, level: str = 'info'):
        """Actualiza la barra de estado"""
        c = self.COLORS
        icons = {'info': '✅', 'warning': '⚠️', 'error': '❌', 'working': '⏳'}
        colors = {'info': c['success'], 'warning': c['warning'],
                  'error': c['error'], 'working': c['accent']}
        icon = icons.get(level, 'ℹ️')
        self.status_bar.config(text=f"{icon} {message}", fg=colors.get(level, c['text_secondary']))
        
    def _create_widgets(self):
        """Crea todos los widgets de la interfaz"""
        # Frame principal
        main_frame = ttk.Frame(self)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Panel lateral izquierdo
        left_panel = ttk.Frame(main_frame, width=300)
        left_panel.pack(side='left', fill='y', padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Panel de contenido principal
        content_panel = ttk.Frame(main_frame)
        content_panel.pack(side='right', fill='both', expand=True)
        
        self._create_left_panel(left_panel)
        self._create_content_panel(content_panel)
        
    def _create_left_panel(self, parent):
        """Crea el panel lateral de selección de modelos"""
        c = self.COLORS
        
        # Título con estilo
        title_frame = tk.Frame(parent, bg=c['sidebar_header'])
        title_frame.pack(fill='x', pady=(0, 5))
        tk.Label(title_frame, text="🔬 MODELOS", 
                font=('Segoe UI', 14, 'bold'),
                bg=c['sidebar_header'], fg=c['text_primary'],
                padx=10, pady=12).pack(fill='x')
        tk.Label(title_frame, text="Selecciona un modelo",
                font=('Segoe UI', 9), bg=c['sidebar_header'],
                fg=c['text_secondary'], padx=10).pack(fill='x', pady=(0, 8))
        
        # Botones de selección de modelos
        models_frame = tk.Frame(parent, bg=c['sidebar'])
        models_frame.pack(fill='both', expand=True, padx=5)
        
        button_configs = [
            ("🧊  Enfriamiento Newton", ModelType.COOLING),
            ("☢️  Decaimiento Radioactivo", ModelType.RADIOACTIVE),
            ("✅  ED Exactas", ModelType.EXACT_DE),
            ("🔄  ED No Exactas", ModelType.NONEXACT_DE),
            ("∫   Cálculo Simbólico", ModelType.SYMBOLIC),
            ("∂   Derivadas Parciales", ModelType.PARTIAL_DERIV)
        ]
        
        self.model_buttons = {}
        for text, model_type in button_configs:
            btn = tk.Button(models_frame, text=text, font=('Segoe UI', 10),
                          command=lambda mt=model_type: self._switch_model(mt),
                          bg=c['btn_normal'], fg=c['text_primary'],
                          activebackground=c['btn_hover'], activeforeground='white',
                          relief='flat', padx=20, pady=10, anchor='w',
                          cursor='hand2', bd=0)
            btn.pack(fill='x', pady=1, padx=2)
            # Hover effects
            btn.bind('<Enter>', lambda e, b=btn: b.config(bg=c['btn_hover']))
            btn.bind('<Leave>', lambda e, b=btn, mt=model_type: 
                     b.config(bg=c['btn_active'] if self.current_model == mt else c['btn_normal']))
            self.model_buttons[model_type] = btn
        
        # Info panel
        info_frame = tk.Frame(parent, bg=c['sidebar'], padx=8, pady=8)
        info_frame.pack(fill='x', pady=(15, 5), padx=5)
        
        tk.Label(info_frame, text="ℹ️ Acerca de",
                font=('Segoe UI', 10, 'bold'), bg=c['sidebar'],
                fg=c['accent']).pack(anchor='w', pady=(0, 5))
        
        info_text = """• Enfriamiento de Newton
• Decaimiento radioactivo
• ED Exactas y No Exactas
• Cálculo simbólico
• Derivadas parciales"""
        
        tk.Label(info_frame, text=info_text,
                font=('Segoe UI', 9), justify='left',
                bg=c['sidebar'], fg=c['text_secondary'],
                padx=5, pady=3).pack(anchor='w')
        
    def _create_content_panel(self, parent):
        """Crea el panel de contenido principal"""
        # Notebook para pestañas
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill='both', expand=True)
        
        # Crear pestañas para cada modelo
        self.tabs = {}
        self.tab_frames = {}
        
        for model_type in ModelType:
            frame = ttk.Frame(self.notebook)
            self.tabs[model_type] = frame
            self.notebook.add(frame, text=model_type.value)
            
            # Inicializar cada pestaña
            init_method = getattr(self, f'_init_{model_type.name.lower()}_tab', None)
            if init_method:
                init_method(frame)
    
    def _switch_model(self, model_type):
        """Cambia al modelo seleccionado"""
        c = self.COLORS
        self.current_model = model_type
        
        # Resaltar botón activo con colores del tema
        for btn_type, btn in self.model_buttons.items():
            if btn_type == model_type:
                btn.config(bg=c['btn_active'], fg='white')
            else:
                btn.config(bg=c['btn_normal'], fg=c['text_primary'])
        
        # Cambiar a la pestaña correspondiente
        tab_id = list(self.tabs.keys()).index(model_type)
        self.notebook.select(tab_id)
        
        self._update_status(f"Modelo activo: {model_type.value}")
    
    def _setup_defaults(self):
        """Configura valores por defecto"""
        self._switch_model(ModelType.COOLING)
    
    # ===== INICIALIZACIÓN DE PESTAÑAS =====
    
    def _init_cooling_tab(self, parent):
        """Inicializa la pestaña de enfriamiento de Newton"""
        c = self.COLORS
        
        # Frame dividido
        left_frame = ttk.Frame(parent)
        left_frame.pack(side='left', fill='y', padx=10, pady=10)
        
        right_frame = ttk.Frame(parent)
        right_frame.pack(side='right', fill='both', expand=True, padx=10, pady=10)
        
        # Controles
        tk.Label(left_frame, text="Parámetros de Enfriamiento", 
                 font=('Segoe UI', 11, 'bold'), bg=c['bg'], fg=c['accent']).pack(anchor='w', pady=(0, 10))
        
        # Variables
        self.T0_var = tk.DoubleVar(value=self.params.T0)
        self.Tenv_var = tk.DoubleVar(value=self.params.Tenv)
        self.k_var = tk.DoubleVar(value=self.params.k)
        self.tmax_c_var = tk.DoubleVar(value=self.params.tmax)
        
        # Campos de entrada
        fields = [
            ("Temperatura inicial T₀ (°C):", self.T0_var),
            ("Temperatura ambiente T_env (°C):", self.Tenv_var),
            ("Constante k (1/minuto):", self.k_var),
            ("Tiempo máximo (minutos):", self.tmax_c_var),
        ]
        
        for label, var in fields:
            frame = ttk.Frame(left_frame)
            frame.pack(fill='x', pady=5)
            tk.Label(frame, text=label, width=25, anchor='w', bg=c['bg'], fg=c['text_primary']).pack(side='left')
            ttk.Entry(frame, textvariable=var, width=15).pack(side='right')
        
        # Botones
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill='x', pady=20)
        
        btn_auto = ttk.Button(btn_frame, text="Calcular Automáticamente",
                   command=self._calculate_auto_cooling)
        btn_auto.pack(fill='x', pady=2)
        
        btn_plot = ttk.Button(btn_frame, text="Graficar Solución",
                   command=self._plot_cooling)
        btn_plot.pack(fill='x', pady=2)
        
        btn_math = ttk.Button(btn_frame, text="Ver Derivación Matemática",
                   command=self._show_cooling_derivation)
        btn_math.pack(fill='x', pady=2)
        
        # Área de gráfica
        self.fig_cooling = Figure(figsize=(8, 6), facecolor=c['bg'])
        self.ax_cooling = self.fig_cooling.add_subplot(111, facecolor=c['content_bg'])
        self.canvas_cooling = FigureCanvasTkAgg(self.fig_cooling, right_frame)
        self.canvas_cooling.get_tk_widget().pack(fill='both', expand=True)
    
    def _init_radioactive_tab(self, parent):
        """Inicializa la pestaña de decaimiento radioactivo"""
        c = self.COLORS
        
        left_frame = ttk.Frame(parent)
        left_frame.pack(side='left', fill='y', padx=10, pady=10)
        
        right_frame = ttk.Frame(parent)
        right_frame.pack(side='right', fill='both', expand=True, padx=10, pady=10)
        
        # Controles
        tk.Label(left_frame, text="Parámetros de Decaimiento", 
                 font=('Segoe UI', 11, 'bold'), bg=c['bg'], fg=c['accent']).pack(anchor='w', pady=(0, 10))
        
        # Variables
        self.N0_var = tk.IntVar(value=self.params.N0)
        self.half_var = tk.DoubleVar(value=self.params.half_life)
        self.tmax_d_var = tk.DoubleVar(value=50.0)
        self.sims_var = tk.IntVar(value=self.params.sim_count)
        
        fields = [
            ("Núcleos iniciales N₀:", self.N0_var),
            ("Vida media t½:", self.half_var),
            ("Tiempo máximo:", self.tmax_d_var),
            ("N° simulaciones:", self.sims_var),
        ]
        
        for label, var in fields:
            frame = ttk.Frame(left_frame)
            frame.pack(fill='x', pady=5)
            tk.Label(frame, text=label, width=20, anchor='w', bg=c['bg'], fg=c['text_primary']).pack(side='left')
            ttk.Entry(frame, textvariable=var, width=15).pack(side='right')
        
        # Botones
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill='x', pady=20)
        
        btn_sim = ttk.Button(btn_frame, text="Ejecutar Simulación",
                   command=self._plot_decay)
        btn_sim.pack(fill='x', pady=2)
        
        btn_comp = ttk.Button(btn_frame, text="Comparar Métodos",
                   command=self._compare_decay_methods)
        btn_comp.pack(fill='x', pady=2)
        
        # Área de gráfica
        self.fig_decay = Figure(figsize=(8, 6), facecolor=c['bg'])
        self.ax_decay = self.fig_decay.add_subplot(111, facecolor=c['content_bg'])
        self.canvas_decay = FigureCanvasTkAgg(self.fig_decay, right_frame)
        self.canvas_decay.get_tk_widget().pack(fill='both', expand=True)
    
    def _init_exact_de_tab(self, parent):
        """Inicializa la pestaña de ED Exactas"""
        c = self.COLORS
        
        # Frame principal dividido
        top_frame = ttk.Frame(parent)
        top_frame.pack(fill='x', padx=10, pady=10)
        
        bottom_frame = ttk.Frame(parent)
        bottom_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Entrada de ecuaciones
        tk.Label(top_frame, text="Ecuación Exacta: M(x,y)dx + N(x,y)dy = 0",
                 font=('Segoe UI', 11, 'bold'), bg=c['bg'], fg=c['accent']).pack(anchor='w', pady=(0, 10))
        
        input_frame = ttk.Frame(top_frame)
        input_frame.pack(fill='x', pady=10)
        
        tk.Label(input_frame, text="M(x,y) =", bg=c['bg'], fg=c['text_primary']).pack(side='left', padx=(0, 5))
        self.M_expr_var = tk.StringVar(value="2*x*y")
        ttk.Entry(input_frame, textvariable=self.M_expr_var, width=20).pack(side='left', padx=(0, 20))
        
        tk.Label(input_frame, text="N(x,y) =", bg=c['bg'], fg=c['text_primary']).pack(side='left', padx=(0, 5))
        self.N_expr_var = tk.StringVar(value="x**2 - y**2")
        ttk.Entry(input_frame, textvariable=self.N_expr_var, width=20).pack(side='left')
        
        # Condiciones iniciales
        cond_frame = ttk.Frame(top_frame)
        cond_frame.pack(fill='x', pady=10)
        
        tk.Label(cond_frame, text="Condición inicial:", bg=c['bg'], fg=c['text_secondary']).pack(side='left', padx=(0, 5))
        tk.Label(cond_frame, text="x₀ =", bg=c['bg'], fg=c['text_primary']).pack(side='left', padx=(0, 5))
        self.x0_var = tk.DoubleVar(value=1.0)
        ttk.Entry(cond_frame, textvariable=self.x0_var, width=8).pack(side='left', padx=(0, 15))
        
        tk.Label(cond_frame, text="y₀ =", bg=c['bg'], fg=c['text_primary']).pack(side='left', padx=(0, 5))
        self.y0_var = tk.DoubleVar(value=2.0)
        ttk.Entry(cond_frame, textvariable=self.y0_var, width=8).pack(side='left')
        
        # Botón de solución
        ttk.Button(top_frame, text="Resolver ED Exacta",
                  command=self._solve_exact_de).pack(pady=10)
        
        # Área de resultados
        self.exact_result_text = scrolledtext.ScrolledText(bottom_frame, 
                                                          wrap=tk.WORD,
                                                          font=('Consolas', 10),
                                                          height=20,
                                                          bg=c['content_bg'],
                                                          fg=c['text_primary'],
                                                          insertbackground='white',
                                                          bd=0,
                                                          padx=8,
                                                          pady=8)
        self.exact_result_text.pack(fill='both', expand=True)
    
    def _init_nonexact_de_tab(self, parent):
        """Inicializa la pestaña de ED No Exactas"""
        c = self.COLORS
        
        # Frame principal dividido
        top_frame = ttk.Frame(parent)
        top_frame.pack(fill='x', padx=10, pady=10)
        
        bottom_frame = ttk.Frame(parent)
        bottom_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Título y descripción
        tk.Label(top_frame, text="Buscar Factor Integrante para ED No Exactas",
                 font=('Segoe UI', 11, 'bold'), bg=c['bg'], fg=c['accent']).pack(anchor='w', pady=(0, 10))
        
        desc_text = "Para ecuaciones de la forma M(x,y)dx + N(x,y)dy = 0 que NO son exactas"
        tk.Label(top_frame, text=desc_text, font=('Segoe UI', 9), bg=c['bg'], fg=c['text_secondary']).pack(anchor='w', pady=(0, 5))
        
        # Frame de entrada
        input_frame = ttk.LabelFrame(top_frame, text="Ingresar ecuación", padding=10)
        input_frame.pack(fill='x', pady=10)
        
        # M(x,y)
        m_frame = ttk.Frame(input_frame)
        m_frame.pack(fill='x', pady=5)
        tk.Label(m_frame, text="M(x,y) =", width=10, anchor='w', bg=c['bg'], fg=c['text_primary']).pack(side='left')
        self.M_nonexact_var = tk.StringVar(value="y")
        ttk.Entry(m_frame, textvariable=self.M_nonexact_var, width=30).pack(side='left', padx=5)
        
        # N(x,y)
        n_frame = ttk.Frame(input_frame)
        n_frame.pack(fill='x', pady=5)
        tk.Label(n_frame, text="N(x,y) =", width=10, anchor='w', bg=c['bg'], fg=c['text_primary']).pack(side='left')
        self.N_nonexact_var = tk.StringVar(value="x + y")
        ttk.Entry(n_frame, textvariable=self.N_nonexact_var, width=30).pack(side='left', padx=5)
        
        # Ejemplos rápidos
        examples_frame = ttk.LabelFrame(top_frame, text="Ejemplos rápidos", padding=10)
        examples_frame.pack(fill='x', pady=10)
        
        examples = [
            ("Ejemplo 1 (NO exacta):", "y**2", "x"),
            ("Ejemplo 2 (NO exacta):", "y", "2*x"),
            ("Ejemplo 3 (NO exacta):", "2*y", "3*x**2"),
        ]
        
        for desc, m_ex, n_ex in examples:
            frame = ttk.Frame(examples_frame)
            frame.pack(fill='x', pady=2)
            tk.Label(frame, text=desc, width=25, anchor='w', bg=c['bg'], fg=c['text_secondary']).pack(side='left')
            btn = tk.Button(frame, text="Usar", 
                      command=lambda m=m_ex, n=n_ex: self._load_example_nonexact(m, n),
                      width=10, font=('Segoe UI', 9), bg=c['btn_normal'], fg=c['text_primary'], relief='flat')
            btn.pack(side='right', padx=5)
            btn.bind('<Enter>', lambda e, b=btn: b.config(bg=c['btn_hover']))
            btn.bind('<Leave>', lambda e, b=btn: b.config(bg=c['btn_normal']))
            
            tk.Label(frame, text=f"M={m_ex}, N={n_ex}", width=20, bg=c['bg'], fg=c['text_primary']).pack(side='right')
        
        # Botón de búsqueda
        ttk.Button(top_frame, text="Encontrar Factor Integrante",
                  command=self._find_integrating_factor).pack(pady=10)
        
        # Área de resultados
        self.nonexact_result_text = scrolledtext.ScrolledText(bottom_frame,
                                                             wrap=tk.WORD,
                                                             font=('Consolas', 10),
                                                             height=20,
                                                             bg=c['content_bg'],
                                                             fg=c['text_primary'],
                                                             insertbackground='white',
                                                             bd=0,
                                                             padx=8,
                                                             pady=8)
        self.nonexact_result_text.pack(fill='both', expand=True)
    
    def _init_symbolic_tab(self, parent):
        """Inicializa la pestaña de cálculo simbólico"""
        c = self.COLORS
        
        # Frame de entrada
        input_frame = ttk.Frame(parent)
        input_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(input_frame, text="Expresión:", font=('Segoe UI', 10), bg=c['bg'], fg=c['text_primary']).pack(side='left', padx=(0, 5))
        self.sym_expr_var = tk.StringVar(value="x**2 + sin(x)")
        ttk.Entry(input_frame, textvariable=self.sym_expr_var, width=30).pack(side='left', padx=(0, 20))
        
        tk.Label(input_frame, text="Variable:", font=('Segoe UI', 10), bg=c['bg'], fg=c['text_primary']).pack(side='left', padx=(0, 5))
        self.sym_var_var = tk.StringVar(value="x")
        ttk.Entry(input_frame, textvariable=self.sym_var_var, width=5).pack(side='left', padx=(0, 20))
        
        # Botones de operaciones
        ops_frame = ttk.Frame(parent)
        ops_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(ops_frame, text="Derivar", 
                  command=lambda: self._symbolic_operation('derivative')).pack(side='left', padx=2)
        ttk.Button(ops_frame, text="Integrar (indefinida)",
                  command=lambda: self._symbolic_operation('integral_indef')).pack(side='left', padx=2)
        ttk.Button(ops_frame, text="Integrar (definida)",
                  command=self._symbolic_definite_integral).pack(side='left', padx=2)
        ttk.Button(ops_frame, text="Simplificar",
                  command=lambda: self._symbolic_operation('simplify')).pack(side='left', padx=2)
        
        # Área de resultados
        result_frame = ttk.Frame(parent)
        result_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.symbolic_result_text = scrolledtext.ScrolledText(result_frame,
                                                            wrap=tk.WORD,
                                                            font=('Consolas', 10),
                                                            height=15,
                                                            bg=c['content_bg'],
                                                            fg=c['text_primary'],
                                                            insertbackground='white',
                                                            bd=0,
                                                            padx=8,
                                                            pady=8)
        self.symbolic_result_text.pack(fill='both', expand=True)
    
    def _init_partial_deriv_tab(self, parent):
        """Inicializa la pestaña de derivadas parciales"""
        c = self.COLORS
        
        # Entrada
        input_frame = ttk.Frame(parent)
        input_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(input_frame, text="f(", font=('Segoe UI', 10), bg=c['bg'], fg=c['text_primary']).pack(side='left')
        self.partial_expr_var = tk.StringVar(value="x**2*y + sin(x*y)")
        ttk.Entry(input_frame, textvariable=self.partial_expr_var, width=25).pack(side='left')
        tk.Label(input_frame, text=") donde variables:", font=('Segoe UI', 10), bg=c['bg'], fg=c['text_primary']).pack(side='left', padx=(5, 5))
        self.partial_vars_var = tk.StringVar(value="x,y")
        ttk.Entry(input_frame, textvariable=self.partial_vars_var, width=10).pack(side='left')
        
        # Opciones
        options_frame = ttk.Frame(parent)
        options_frame.pack(fill='x', padx=10, pady=5)
        
        self.partial_order_var = tk.IntVar(value=1)
        ttk.Radiobutton(options_frame, text="Primer orden", 
                       variable=self.partial_order_var, value=1).pack(side='left', padx=10)
        ttk.Radiobutton(options_frame, text="Segundo orden", 
                       variable=self.partial_order_var, value=2).pack(side='left', padx=10)
        
        ttk.Button(options_frame, text="Calcular Derivadas Parciales",
                  command=self._calculate_partial_derivatives).pack(side='right', padx=10)
        
        # Área de resultados
        result_frame = ttk.Frame(parent)
        result_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.partial_result_text = scrolledtext.ScrolledText(result_frame,
                                                           wrap=tk.WORD,
                                                           font=('Consolas', 10),
                                                           height=15,
                                                           bg=c['content_bg'],
                                                           fg=c['text_primary'],
                                                           insertbackground='white',
                                                           bd=0,
                                                           padx=8,
                                                           pady=8)
        self.partial_result_text.pack(fill='both', expand=True)
    
    # ===== MÉTODOS DE CÁLCULO =====
    
    def _calculate_auto_cooling(self):
        """Calcula parámetros automáticos para enfriamiento"""
        try:
            T0 = self.T0_var.get()
            Tenv = self.Tenv_var.get()
            
            if T0 == Tenv:
                self._update_status("T₀ y T_env no pueden ser iguales", 'warning')
                messagebox.showwarning("Aviso", "T₀ y T_env no pueden ser iguales")
                return
            
            auto_k = calculate_auto_k(T0, Tenv)
            auto_tmax = calculate_auto_tmax(T0, Tenv, auto_k)
            
            self.k_var.set(round(auto_k, 4))
            self.tmax_c_var.set(round(auto_tmax, 1))
            
            self._update_status(f"Parámetros calculados: k={auto_k:.4f}, t_max={auto_tmax:.1f}")
            messagebox.showinfo("Cálculo Automático", 
                              f"k = {auto_k:.4f}\nt_max = {auto_tmax:.1f}")
            
        except Exception as e:
            self._update_status(f"Error: {e}", 'error')
            messagebox.showerror("Error", str(e))
    
    def _plot_cooling(self):
        """Grafica la solución de enfriamiento con estilo profesional"""
        try:
            T0 = self.T0_var.get()
            Tenv = self.Tenv_var.get()
            k = self.k_var.get()
            tmax = self.tmax_c_var.get()
            
            # Validación
            if k <= 0:
                self._update_status("La constante k debe ser positiva", 'warning')
                messagebox.showwarning("Aviso", "La constante k debe ser > 0")
                return
            if tmax <= 0:
                self._update_status("El tiempo máximo debe ser positivo", 'warning')
                messagebox.showwarning("Aviso", "El tiempo máximo debe ser > 0")
                return
            
            self._update_status("Calculando solución de enfriamiento...", 'working')
            
            t = np.linspace(0, tmax, 400)
            T_analytic = PhysicsModels.newton_analytic(T0, Tenv, k, t)
            T_euler = PhysicsModels.newton_euler(T0, Tenv, k, t)
            
            # Estilo oscuro para las gráficas
            self.fig_cooling.patch.set_facecolor('#1a1a2e')
            self.ax_cooling.set_facecolor('#0f0f23')
            self.ax_cooling.clear()
            
            self.ax_cooling.plot(t, T_analytic, color='#e94560', label='Solución Analítica', linewidth=2.5)
            self.ax_cooling.plot(t, T_euler, color='#00d2ff', linestyle='--', 
                               label='Método de Euler', linewidth=1.5, alpha=0.8)
            self.ax_cooling.axhline(y=Tenv, color='#f39c12', linestyle=':', 
                                   alpha=0.6, label=f'T_env = {Tenv}°C')
            
            self.ax_cooling.set_xlabel('Tiempo (min)', color='#eaeaea', fontsize=11)
            self.ax_cooling.set_ylabel('Temperatura (°C)', color='#eaeaea', fontsize=11)
            self.ax_cooling.set_title(f'Ley de Enfriamiento de Newton\nT₀={T0}°C, T_env={Tenv}°C, k={k:.4f}',
                                     color='#eaeaea', fontsize=12, fontweight='bold')
            self.ax_cooling.legend(facecolor='#16213e', edgecolor='#3a3a5a', 
                                  labelcolor='#eaeaea', fontsize=9)
            self.ax_cooling.grid(True, alpha=0.15, color='#4a4a6a')
            self.ax_cooling.tick_params(colors='#a0a0b0')
            for spine in self.ax_cooling.spines.values():
                spine.set_color('#3a3a5a')
            
            self.fig_cooling.tight_layout()
            self.canvas_cooling.draw()
            
            self._update_status(f"Gráfica de enfriamiento generada exitosamente")
            
        except Exception as e:
            self._update_status(f"Error al graficar: {e}", 'error')
            messagebox.showerror("Error al graficar", str(e))
    
    def _plot_decay(self):
        """Grafica decaimiento radioactivo con estilo profesional"""
        try:
            N0 = self.N0_var.get()
            half = self.half_var.get()
            tmax = self.tmax_d_var.get()
            sims = self.sims_var.get()
            
            # Validación
            if N0 <= 0:
                self._update_status("N₀ debe ser positivo", 'warning')
                return
            if half <= 0:
                self._update_status("La vida media debe ser positiva", 'warning')
                return
            if sims < 1 or sims > 20:
                self._update_status("N° simulaciones debe estar entre 1 y 20", 'warning')
                return
            
            self._update_status("Ejecutando simulaciones Monte Carlo...", 'working')
            
            t = np.linspace(0, tmax, 201)
            N_analytic = PhysicsModels.radioactive_analytic(N0, half, t)
            
            # Estilo oscuro
            self.fig_decay.patch.set_facecolor('#1a1a2e')
            self.ax_decay.set_facecolor('#0f0f23')
            self.ax_decay.clear()
            
            self.ax_decay.plot(t, N_analytic, color='#e94560', 
                              label='Solución Analítica', linewidth=3, alpha=0.9)
            
            # Simulaciones estocásticas con colores variados
            sim_colors = ['#00d2ff', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c',
                         '#e67e22', '#3498db', '#e74c3c', '#27ae60', '#8e44ad']
            rng = np.random.default_rng()
            for i in range(sims):
                color = sim_colors[i % len(sim_colors)]
                counts = PhysicsModels.radioactive_stochastic(N0, half, t, rng=rng)
                self.ax_decay.step(t, counts, where='post', alpha=0.5, 
                                  linewidth=1, color=color)
            
            self.ax_decay.set_xlabel('Tiempo', color='#eaeaea', fontsize=11)
            self.ax_decay.set_ylabel('Núcleos Restantes', color='#eaeaea', fontsize=11)
            self.ax_decay.set_title(f'Decaimiento Radioactivo\nN₀={N0}, t½={half}',
                                   color='#eaeaea', fontsize=12, fontweight='bold')
            self.ax_decay.legend(facecolor='#16213e', edgecolor='#3a3a5a',
                                labelcolor='#eaeaea', fontsize=9)
            self.ax_decay.grid(True, alpha=0.15, color='#4a4a6a')
            self.ax_decay.tick_params(colors='#a0a0b0')
            for spine in self.ax_decay.spines.values():
                spine.set_color('#3a3a5a')
            self.ax_decay.set_ylim(bottom=0)
            self.fig_decay.tight_layout()
            self.canvas_decay.draw()
            
            self._update_status(f"Decaimiento: {sims} simulaciones completadas")
            
        except Exception as e:
            self._update_status(f"Error: {e}", 'error')
            messagebox.showerror("Error al graficar", str(e))
    
    def _solve_exact_de(self):
        """Resuelve una ED exacta"""
        try:
            M_expr = self.M_expr_var.get()
            N_expr = self.N_expr_var.get()
            x0 = self.x0_var.get()
            y0 = self.y0_var.get()
            
            result = PhysicsModels.exact_de_solution(M_expr, N_expr, x0, y0)
            
            # Mostrar resultados
            self.exact_result_text.delete(1.0, tk.END)
            
            if result['exact']:
                self.exact_result_text.insert(tk.END, "✅ ECUACIÓN EXACTA\n\n")
                for step in result['steps']:
                    self.exact_result_text.insert(tk.END, f"• {step}\n")
                
                self.exact_result_text.insert(tk.END, f"\n📝 Solución:\n{result['solution']}\n")
                
                if 'F_at_point' in result:
                    self.exact_result_text.insert(tk.END, f"\n🔍 F({x0},{y0}) = {result['F_at_point']}\n")
                
                if 'verification_x' in result and 'verification_y' in result:
                    self.exact_result_text.insert(tk.END, f"\n✅ Verificación:\n")
                    self.exact_result_text.insert(tk.END, f"  dF/dx - M = {result['verification_x']}\n")
                    self.exact_result_text.insert(tk.END, f"  dF/dy - N = {result['verification_y']}\n")
                    
            else:
                self.exact_result_text.insert(tk.END, "❌ La ecuación NO es exacta\n\n")
                for step in result['steps']:
                    self.exact_result_text.insert(tk.END, f"• {step}\n")
                
                self.exact_result_text.insert(tk.END, "\n💡 Intenta buscar un factor integrante en la pestaña 'ED No Exactas'\n")
                
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo resolver la ED:\n{str(e)}")
    
    def _load_example_nonexact(self, M_expr: str, N_expr: str):
        """Carga un ejemplo en la pestaña de ED No Exactas"""
        self.M_nonexact_var.set(M_expr)
        self.N_nonexact_var.set(N_expr)
    
    def _find_integrating_factor(self):
        """Busca factor integrante para ED no exactas"""
        try:
            M_expr = self.M_nonexact_var.get()
            N_expr = self.N_nonexact_var.get()
            
            result = PhysicsModels.find_integrating_factor(M_expr, N_expr)
            
            # Mostrar resultados
            self.nonexact_result_text.delete(1.0, tk.END)
            
            if not result['needs_factor']:
                self.nonexact_result_text.insert(tk.END, "✅ La ecuación YA ES EXACTA\n\n")
                for step in result['steps']:
                    self.nonexact_result_text.insert(tk.END, f"{step}\n")
                
                self.nonexact_result_text.insert(tk.END, "\n" + "="*50 + "\n")
                self.nonexact_result_text.insert(tk.END, "\nℹ️ INFORMACIÓN:\n")
                self.nonexact_result_text.insert(tk.END, f"Como ∂M/∂y = ∂N/∂x, la ecuación ya es exacta.\n")
                self.nonexact_result_text.insert(tk.END, f"Puedes resolverla directamente en la pestaña 'ED Exactas'.\n")
                self.nonexact_result_text.insert(tk.END, "\n📝 Ejemplos de ecuaciones NO exactas para probar:\n")
                self.nonexact_result_text.insert(tk.END, "1. M = y²,      N = x      → Factor: μ(x) = 1/x²\n")
                self.nonexact_result_text.insert(tk.END, "2. M = y,       N = 2x     → Factor: μ(y) = 1/y\n")
                self.nonexact_result_text.insert(tk.END, "3. M = 2y,      N = 3x²    → Factor: μ(x) = x\n")
                self.nonexact_result_text.insert(tk.END, "4. M = x² + y,  N = x      → No tiene factor simple\n")
                
            elif result['factor']:
                self.nonexact_result_text.insert(tk.END, f"✅ FACTOR INTEGRANTE ENCONTRADO\n\n")
                for step in result['steps']:
                    self.nonexact_result_text.insert(tk.END, f"{step}\n")
                
                self.nonexact_result_text.insert(tk.END, "\n" + "="*50 + "\n")
                self.nonexact_result_text.insert(tk.END, f"\n📋 RESUMEN:\n")
                self.nonexact_result_text.insert(tk.END, f"• Tipo: {result['type']}\n")
                self.nonexact_result_text.insert(tk.END, f"• Factor integrante: μ = {result['factor']}\n")
                
                self.nonexact_result_text.insert(tk.END, f"\n🔧 CÓMO USAR EL FACTOR INTEGRANTE:\n")
                self.nonexact_result_text.insert(tk.END, f"1. Multiplica la ecuación original por μ\n")
                self.nonexact_var = result['factor']  # Guardar para posible uso futuro
                
                # Mostrar la nueva ecuación exacta
                self.nonexact_result_text.insert(tk.END, f"2. Nueva ecuación exacta: μM dx + μN dy = 0\n")
                self.nonexact_result_text.insert(tk.END, f"3. Resuelve en la pestaña 'ED Exactas' con:\n")
                self.nonexact_result_text.insert(tk.END, f"   M_nueva = {result['factor']} * ({M_expr})\n")
                self.nonexact_result_text.insert(tk.END, f"   N_nueva = {result['factor']} * ({N_expr})\n")
                
            else:
                self.nonexact_result_text.insert(tk.END, "❌ No se encontró factor integrante simple\n\n")
                for step in result['steps']:
                    self.nonexact_result_text.insert(tk.END, f"{step}\n")
                
                self.nonexact_result_text.insert(tk.END, "\n" + "="*50 + "\n")
                self.nonexact_result_text.insert(tk.END, f"\nℹ️ EXPLICACIÓN:\n")
                self.nonexact_result_text.insert(tk.END, f"No se pudo encontrar un factor integrante que sea:\n")
                self.nonexact_result_text.insert(tk.END, f"• Solo función de x: μ(x) = exp(∫[(∂M/∂y - ∂N/∂x)/N] dx)\n")
                self.nonexact_result_text.insert(tk.END, f"• Solo función de y: μ(y) = exp(∫[(∂N/∂x - ∂M/∂y)/M] dy)\n")
                self.nonexact_result_text.insert(tk.END, f"\n💡 SUGERENCIAS:\n")
                self.nonexact_result_text.insert(tk.END, f"1. Intenta otros métodos (μ(x,y), agrupación, etc.)\n")
                self.nonexact_result_text.insert(tk.END, f"2. Verifica si la ecuación es de variables separables\n")
                self.nonexact_result_text.insert(tk.END, f"3. Prueba con otros ejemplos de la lista arriba\n")
                
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo buscar factor integrante:\n{str(e)}")
    
    def _symbolic_operation(self, op_type):
        """Realiza operación simbólico"""
        try:
            expr = self.sym_expr_var.get()
            var = self.sym_var_var.get()
            
            if op_type == 'derivative':
                result = PhysicsModels.symbolic_derivative(expr, var)
                output = f"Derivada de {expr} respecto a {var}:\n"
                output += f"  d/d{var} = {result['derivative']}\n"
                output += f"  Simplificado: {result['simplified']}\n"
                output += f"  LaTeX: {result['latex']}"
                
            elif op_type == 'integral_indef':
                result = PhysicsModels.symbolic_integral(expr, var, definite=False)
                if result['success']:
                    output = f"Integral de {expr} respecto a {var}:\n"
                    output += f"  ∫ {expr} d{var} = {result['integral']}\n"
                    output += f"  LaTeX: {result['latex']}"
                else:
                    output = f"Error al calcular la integral:\n{result['error']}"
                
            elif op_type == 'simplify':
                x = sp.symbols(var)
                expr_sym = sp.sympify(expr)
                simplified = sp.simplify(expr_sym)
                output = f"Simplificación de {expr}:\n"
                output += f"  Resultado: {simplified}\n"
                output += f"  LaTeX: {sp.latex(simplified)}"
            
            self.symbolic_result_text.delete(1.0, tk.END)
            self.symbolic_result_text.insert(tk.END, output)
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo realizar la operación:\n{str(e)}")
    
    def _symbolic_definite_integral(self):
        """Ventana para integral definida"""
        dialog = tk.Toplevel(self)
        dialog.title("Integral Definida")
        dialog.geometry("300x150")
        
        tk.Label(dialog, text="Límite inferior a:").pack(pady=(10, 0))
        a_var = tk.DoubleVar(value=0.0)
        ttk.Entry(dialog, textvariable=a_var, width=20).pack(pady=5)
        
        tk.Label(dialog, text="Límite superior b:").pack(pady=(5, 0))
        b_var = tk.DoubleVar(value=1.0)
        ttk.Entry(dialog, textvariable=b_var, width=20).pack(pady=5)
        
        def calculate():
            try:
                expr = self.sym_expr_var.get()
                var = self.sym_var_var.get()
                a = a_var.get()
                b = b_var.get()
                
                result = PhysicsModels.symbolic_integral(expr, var, definite=True, limits=(a, b))
                
                if result['success']:
                    output = f"Integral definida de {expr} respecto a {var}:\n"
                    output += f"  ∫_{{{a}}}^{{{b}}} {expr} d{var} = {result['integral']}\n"
                    output += f"  LaTeX: {result['latex']}"
                else:
                    output = f"Error al calcular la integral definida:\n{result['error']}"
                
                self.symbolic_result_text.delete(1.0, tk.END)
                self.symbolic_result_text.insert(tk.END, output)
                
                dialog.destroy()
                
            except Exception as e:
                messagebox.showerror("Error", str(e), parent=dialog)
        
        ttk.Button(dialog, text="Calcular", command=calculate).pack(pady=10)
    
    def _calculate_partial_derivatives(self):
        """Calcula derivadas parciales"""
        try:
            expr = self.partial_expr_var.get()
            vars_str = self.partial_vars_var.get()
            order = self.partial_order_var.get()
            
            result = PhysicsModels.partial_derivatives(expr, vars_str, order)
            
            # Mostrar resultados
            self.partial_result_text.delete(1.0, tk.END)
            
            if result['success']:
                self.partial_result_text.insert(tk.END, f"Función: f({vars_str}) = {expr}\n")
                self.partial_result_text.insert(tk.END, f"Orden: {order}\n")
                self.partial_result_text.insert(tk.END, "="*50 + "\n\n")
                
                for r in result['results']:
                    if order == 1:
                        self.partial_result_text.insert(tk.END, f"∂f/∂{r['variable']} = {r['derivative']}\n")
                        self.partial_result_text.insert(tk.END, f"  LaTeX: {r['latex']}\n\n")
                    else:
                        self.partial_result_text.insert(tk.END, f"{r['variables']} = {r['derivative']}\n")
                        self.partial_result_text.insert(tk.END, f"  LaTeX: {r['latex']}\n\n")
            else:
                self.partial_result_text.insert(tk.END, f"Error: {result.get('error', 'Error desconocido')}")
                    
        except Exception as e:
            messagebox.showerror("Error", f"No se pudieron calcular las derivadas parciales:\n{str(e)}")
    
    def _show_cooling_derivation(self):
        """Muestra la derivación matemática de la ley de enfriamiento"""
        c = self.COLORS
        dialog = tk.Toplevel(self)
        dialog.title("Derivación Matemática - Ley de Enfriamiento")
        dialog.geometry("600x440")
        dialog.configure(bg=c['bg'])
        
        text = scrolledtext.ScrolledText(dialog, wrap=tk.WORD, font=('Segoe UI', 10),
                                         bg=c['content_bg'], fg=c['text_primary'],
                                         insertbackground='white', bd=0, padx=8, pady=8)
        text.pack(fill='both', expand=True, padx=10, pady=10)
        
        derivation_text = """DERIVACIÓN DE LA LEY DE ENFRIAMIENTO DE NEWTON
1. Ley física básica:
   La tasa de cambio de temperatura es proporcional a la diferencia
   entre la temperatura del objeto y la temperatura ambiente.
   
   dT/dt = -k(T - T_env)
   
   donde:
   T = temperatura del objeto
   T_env = temperatura ambiente (constante)
   k = constante de proporcionalidad positiva
   
2. Separación de variables:
   dT/(T - T_env) = -k dt
   
3. Integración:
   ∫ dT/(T - T_env) = -k ∫ dt
   
4. Solución:
   ln|T - T_env| = -kt + C
   
5. Aplicando exponencial:
   T - T_env = e^{-kt + C} = A e^{-kt}
   
6. Condición inicial T(0) = T₀:
   T₀ - T_env = A e^{0} = A
   
7. Solución final:
   T(t) = T_env + (T₀ - T_env) e^{-kt}
   
Esta es la solución analítica que utilizamos en la aplicación."""
        
        text.insert(1.0, derivation_text)
        text.config(state='disabled')
        
        ttk.Button(dialog, text="Cerrar", command=dialog.destroy).pack(pady=10)
    
    def _compare_decay_methods(self):
        """Compara diferentes métodos de resolución de decaimiento"""
        c = self.COLORS
        dialog = tk.Toplevel(self)
        dialog.title("Comparación de Métodos - Decaimiento")
        dialog.geometry("550x340")
        dialog.configure(bg=c['bg'])
        
        text = scrolledtext.ScrolledText(dialog, wrap=tk.WORD, font=('Consolas', 9),
                                         bg=c['content_bg'], fg=c['text_primary'],
                                         insertbackground='white', bd=0, padx=8, pady=8)
        text.pack(fill='both', expand=True, padx=10, pady=10)
        
        try:
            N0 = self.N0_var.get()
            half = self.half_var.get()
            tmax = self.tmax_d_var.get()
            
            t = np.linspace(0, tmax, 5)
            analytic = PhysicsModels.radioactive_analytic(N0, half, t)
            
            comparison = f"COMPARACIÓN DE MÉTODOS - DECAIMIENTO RADIOACTIVO\n"
            comparison += f"N₀ = {N0}, t½ = {half}, t_max = {tmax}\n"
            comparison += "="*50 + "\n\n"
            comparison += "Tiempo (t) | Solución Analítica | Simulación 1 | Simulación 2\n"
            comparison += "-"*60 + "\n"
            
            rng = np.random.default_rng(42)
            sim1 = PhysicsModels.radioactive_stochastic(N0, half, t, rng=rng)
            sim2 = PhysicsModels.radioactive_stochastic(N0, half, t, rng=np.random.default_rng(123))
            
            for i in range(len(t)):
                comparison += f"{t[i]:8.2f} | {analytic[i]:17.2f} | {sim1[i]:12d} | {sim2[i]:12d}\n"
            
            comparison += "\nCONCLUSIÓN:\n"
            comparison += "• La solución analítica es determinista y suave\n"
            comparison += "• Las simulaciones Monte Carlo son estocásticas\n"
            comparison += "• A mayor N₀, las simulaciones se acercan a la solución analítica\n"
            
            text.insert(1.0, comparison)
            
        except Exception as e:
            text.insert(1.0, f"Error: {str(e)}")
        
        text.config(state='disabled')
        
        ttk.Button(dialog, text="Cerrar", command=dialog.destroy).pack(pady=10)

# ===== PUNTO DE ENTRADA PRINCIPAL =====
def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Aplicación Integral de Modelos Físicos y Matemáticos')
    parser.add_argument('--model', type=str, default='cooling',
                       help='Modelo inicial (cooling, radioactive, exact, etc.)')
    
    args = parser.parse_args()
    
    try:
        app = IntegratedMathApp()
        app.mainloop()
    except Exception as e:
        print(f"Error al ejecutar la aplicación: {e}", file=sys.stderr)
        messagebox.showerror("Error Fatal", f"No se pudo iniciar la aplicación:\n{str(e)}")

if __name__ == '__main__':
    main()