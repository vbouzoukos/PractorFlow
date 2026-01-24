"""
Calculator tool for mathematical operations.

Provides safe expression evaluation with support for
basic arithmetic, math functions, and unit conversions.
"""

import ast
import math
import operator
from typing import Any, Dict, List, Optional

from practorflow.llm.tools.base import BaseTool, ToolParameter, ToolResult
from practorflow.logger.logger import get_logger
from practorflow.settings.app_settings import appConfiguration

logger = get_logger("tool", level=appConfiguration.LoggerConfiguration.ToolLevel)


class CalculatorTool(BaseTool):
    """
    Calculator tool for safe mathematical expression evaluation.

    Supports:
    - Basic arithmetic: +, -, *, /, //, %, **
    - Math functions: sin, cos, tan, sqrt, log, exp, abs, round, etc.
    - Constants: pi, e
    - Unit conversions
    """

    SAFE_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    SAFE_FUNCTIONS = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "atan2": math.atan2,
        "sinh": math.sinh,
        "cosh": math.cosh,
        "tanh": math.tanh,
        "log": math.log,
        "log10": math.log10,
        "log2": math.log2,
        "exp": math.exp,
        "floor": math.floor,
        "ceil": math.ceil,
        "degrees": math.degrees,
        "radians": math.radians,
        "factorial": math.factorial,
        "gcd": math.gcd,
    }

    CONSTANTS = {
        "pi": math.pi,
        "e": math.e,
        "tau": math.tau,
        "inf": math.inf,
    }

    UNIT_CONVERSIONS = {
        "length": {
            "m": 1.0,
            "km": 1000.0,
            "cm": 0.01,
            "mm": 0.001,
            "mi": 1609.344,
            "yd": 0.9144,
            "ft": 0.3048,
            "in": 0.0254,
        },
        "weight": {
            "kg": 1.0,
            "g": 0.001,
            "mg": 0.000001,
            "lb": 0.453592,
            "oz": 0.0283495,
        },
        "temperature": {
            "c": "celsius",
            "f": "fahrenheit",
            "k": "kelvin",
        },
        "time": {
            "s": 1.0,
            "ms": 0.001,
            "min": 60.0,
            "h": 3600.0,
            "d": 86400.0,
        },
        "data": {
            "b": 1.0,
            "kb": 1024.0,
            "mb": 1048576.0,
            "gb": 1073741824.0,
            "tb": 1099511627776.0,
        },
    }

    def __init__(self, max_expression_length: int = 1000):
        """
        Initialize calculator tool.

        Args:
            max_expression_length: Maximum expression length in characters.
        """
        self._max_expression_length = max_expression_length

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return (
            "Perform mathematical calculations and unit conversions. "
            "Supports arithmetic operations (+, -, *, /, **, %), "
            "math functions (sqrt, sin, cos, log, etc.), "
            "constants (pi, e), and unit conversions (length, weight, temperature, time, data)."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="expression",
                type="string",
                description="Mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)', 'sin(pi/2)')",
                required=False,
            ),
            ToolParameter(
                name="convert",
                type="object",
                description="Unit conversion: {value: number, from: unit, to: unit} (e.g., {value: 100, from: 'cm', to: 'm'})",
                required=False,
            ),
        ]

    def _safe_eval(self, expression: str) -> float:
        """
        Safely evaluate a mathematical expression.

        Args:
            expression: Mathematical expression string.

        Returns:
            Evaluation result.

        Raises:
            ValueError: If expression is invalid or unsafe.
        """
        if len(expression) > self._max_expression_length:
            raise ValueError(f"Expression exceeds maximum length of {self._max_expression_length}")

        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as e:
            raise ValueError(f"Invalid expression syntax: {e}")

        return self._eval_node(tree.body)

    def _eval_node(self, node: ast.AST) -> Any:
        """
        Recursively evaluate an AST node.

        Args:
            node: AST node to evaluate.

        Returns:
            Evaluation result.

        Raises:
            ValueError: If node type is not allowed.
        """
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant type: {type(node.value)}")

        elif isinstance(node, ast.Name):
            name = node.id.lower()
            if name in self.CONSTANTS:
                return self.CONSTANTS[name]
            raise ValueError(f"Unknown constant: {node.id}")

        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in self.SAFE_OPERATORS:
                raise ValueError(f"Unsupported operator: {op_type.__name__}")

            left = self._eval_node(node.left)
            right = self._eval_node(node.right)

            if op_type == ast.Pow and right > 1000:
                raise ValueError("Exponent too large")

            return self.SAFE_OPERATORS[op_type](left, right)

        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in self.SAFE_OPERATORS:
                raise ValueError(f"Unsupported unary operator: {op_type.__name__}")

            operand = self._eval_node(node.operand)
            return self.SAFE_OPERATORS[op_type](operand)

        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only direct function calls are allowed")

            func_name = node.func.id.lower()
            if func_name not in self.SAFE_FUNCTIONS:
                raise ValueError(f"Unknown function: {node.func.id}")

            args = [self._eval_node(arg) for arg in node.args]
            return self.SAFE_FUNCTIONS[func_name](*args)

        elif isinstance(node, ast.List):
            return [self._eval_node(elem) for elem in node.elts]

        elif isinstance(node, ast.Tuple):
            return tuple(self._eval_node(elem) for elem in node.elts)

        else:
            raise ValueError(f"Unsupported expression type: {type(node).__name__}")

    def _convert_temperature(self, value: float, from_unit: str, to_unit: str) -> float:
        """
        Convert temperature between Celsius, Fahrenheit, and Kelvin.

        Args:
            value: Temperature value.
            from_unit: Source unit (c, f, k).
            to_unit: Target unit (c, f, k).

        Returns:
            Converted temperature.
        """
        if from_unit == "c":
            celsius = value
        elif from_unit == "f":
            celsius = (value - 32) * 5 / 9
        elif from_unit == "k":
            celsius = value - 273.15
        else:
            raise ValueError(f"Unknown temperature unit: {from_unit}")

        if to_unit == "c":
            return celsius
        elif to_unit == "f":
            return celsius * 9 / 5 + 32
        elif to_unit == "k":
            return celsius + 273.15
        else:
            raise ValueError(f"Unknown temperature unit: {to_unit}")

    def _convert_units(self, value: float, from_unit: str, to_unit: str) -> float:
        """
        Convert between units.

        Args:
            value: Numeric value to convert.
            from_unit: Source unit.
            to_unit: Target unit.

        Returns:
            Converted value.

        Raises:
            ValueError: If units are unknown or incompatible.
        """
        from_unit = from_unit.lower()
        to_unit = to_unit.lower()

        if from_unit == to_unit:
            return value

        for category, units in self.UNIT_CONVERSIONS.items():
            if from_unit in units and to_unit in units:
                if category == "temperature":
                    return self._convert_temperature(value, from_unit, to_unit)

                base_value = value * units[from_unit]
                return base_value / units[to_unit]

        raise ValueError(f"Cannot convert from '{from_unit}' to '{to_unit}'")

    def validate_parameters(self, **kwargs) -> Optional[str]:
        """
        Validate that at least one operation is specified.

        Args:
            **kwargs: Parameters to validate.

        Returns:
            Error message if validation fails, None if valid.
        """
        base_error = super().validate_parameters(**kwargs)
        if base_error:
            return base_error

        expression = kwargs.get("expression")
        convert = kwargs.get("convert")

        if not expression and not convert:
            return "Either 'expression' or 'convert' parameter is required"

        if convert:
            if not isinstance(convert, dict):
                return "'convert' must be an object with 'value', 'from', and 'to' fields"
            if "value" not in convert or "from" not in convert or "to" not in convert:
                return "'convert' requires 'value', 'from', and 'to' fields"

        return None

    def execute(self, **kwargs) -> ToolResult:
        """
        Execute calculation or unit conversion.

        Args:
            expression: Mathematical expression to evaluate.
            convert: Unit conversion specification.

        Returns:
            ToolResult with calculation result or error.
        """
        expression = kwargs.get("expression")
        convert = kwargs.get("convert")

        try:
            if expression:
                logger.debug(f"[Calculator] Evaluating: {expression}")

                result = self._safe_eval(expression)

                if isinstance(result, float):
                    if result.is_integer():
                        result = int(result)
                    else:
                        result = round(result, 10)

                return ToolResult(
                    success=True,
                    data=result,
                    metadata={
                        "operation": "evaluate",
                        "expression": expression,
                        "result_type": type(result).__name__,
                    },
                )

            elif convert:
                value = convert.get("value")
                from_unit = convert.get("from")
                to_unit = convert.get("to")

                logger.debug(f"[Calculator] Converting: {value} {from_unit} to {to_unit}")

                result = self._convert_units(float(value), from_unit, to_unit)

                if isinstance(result, float) and result.is_integer():
                    result = int(result)
                else:
                    result = round(result, 10)

                return ToolResult(
                    success=True,
                    data=result,
                    metadata={
                        "operation": "convert",
                        "value": value,
                        "from": from_unit,
                        "to": to_unit,
                    },
                )

        except ValueError as e:
            logger.error(f"[Calculator] Validation error: {e}")
            return ToolResult(success=False, error=str(e))
        except ZeroDivisionError:
            logger.error("[Calculator] Division by zero")
            return ToolResult(success=False, error="Division by zero")
        except OverflowError:
            logger.error("[Calculator] Overflow error")
            return ToolResult(success=False, error="Result too large")
        except Exception as e:
            logger.error(f"[Calculator] Error: {e}")
            return ToolResult(success=False, error=f"Calculation failed: {str(e)}")