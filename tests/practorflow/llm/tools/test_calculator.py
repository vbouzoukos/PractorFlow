import math
import ast
import pytest

from practorflow.llm.tools.calculator import CalculatorTool


class TestCalculatorTool:
    def test_name(self):
        tool = CalculatorTool()
        assert tool.name == "calculator"

    def test_description_non_empty(self):
        tool = CalculatorTool()
        assert isinstance(tool.description, str)
        assert len(tool.description) > 0

    def test_parameters(self):
        tool = CalculatorTool()
        params = tool.parameters
        names = {p.name for p in params}
        assert names == {"expression", "convert"}

    def test_validate_missing_all(self):
        tool = CalculatorTool()
        error = tool.validate_parameters()
        assert error == "Either 'expression' or 'convert' parameter is required"

    def test_validate_expression_only(self):
        tool = CalculatorTool()
        assert tool.validate_parameters(expression="2+2") is None

    def test_validate_convert_only(self):
        tool = CalculatorTool()
        assert tool.validate_parameters(convert={"value": 1, "from": "m", "to": "cm"}) is None

    def test_validate_convert_not_dict(self):
        tool = CalculatorTool()
        error = tool.validate_parameters(convert="bad")
        assert "'convert' must be an object" in error

    def test_validate_convert_missing_fields(self):
        tool = CalculatorTool()
        error = tool.validate_parameters(convert={"value": 1})
        assert "'convert' requires 'value', 'from', and 'to' fields" in error

    def test_simple_arithmetic(self):
        tool = CalculatorTool()
        assert tool._safe_eval("2 + 3 * 4") == 14

    def test_constants(self):
        tool = CalculatorTool()
        assert tool._safe_eval("pi") == math.pi

    def test_unary_ops(self):
        tool = CalculatorTool()
        assert tool._safe_eval("-5") == -5
        assert tool._safe_eval("+5") == 5

    def test_functions(self):
        tool = CalculatorTool()
        assert tool._safe_eval("sqrt(16)") == 4
        assert tool._safe_eval("sin(pi/2)") == 1.0

    def test_list_and_tuple(self):
        tool = CalculatorTool()
        assert tool._safe_eval("[1,2,3]") == [1, 2, 3]
        assert tool._safe_eval("(1,2)") == (1, 2)

    def test_invalid_syntax(self):
        tool = CalculatorTool()
        with pytest.raises(ValueError):
            tool._safe_eval("2 +")

    def test_expression_too_long(self):
        tool = CalculatorTool(max_expression_length=3)
        with pytest.raises(ValueError):
            tool._safe_eval("1234")

    def test_unknown_constant(self):
        tool = CalculatorTool()
        with pytest.raises(ValueError):
            tool._safe_eval("unknown")

    def test_unsupported_operator(self):
        tool = CalculatorTool()
        with pytest.raises(ValueError):
            tool._safe_eval("1 << 2")

    def test_exponent_too_large(self):
        tool = CalculatorTool()
        with pytest.raises(ValueError):
            tool._safe_eval("2 ** 1001")

    def test_unknown_function(self):
        tool = CalculatorTool()
        with pytest.raises(ValueError):
            tool._safe_eval("evil(1)")

    def test_non_direct_call(self):
        tool = CalculatorTool()
        tree = ast.parse("math.sqrt(4)", mode="eval")
        with pytest.raises(ValueError):
            tool._eval_node(tree.body)

    def test_unsupported_node(self):
        tool = CalculatorTool()
        node = ast.parse("{1:2}", mode="eval").body
        with pytest.raises(ValueError):
            tool._eval_node(node)

    def test_same_unit(self):
        tool = CalculatorTool()
        assert tool._convert_units(10, "m", "m") == 10

    def test_length_conversion(self):
        tool = CalculatorTool()
        assert tool._convert_units(100, "cm", "m") == 1

    def test_weight_conversion(self):
        tool = CalculatorTool()
        assert tool._convert_units(1000, "g", "kg") == 1

    def test_time_conversion(self):
        tool = CalculatorTool()
        assert tool._convert_units(60, "s", "min") == 1

    def test_data_conversion(self):
        tool = CalculatorTool()
        assert tool._convert_units(1024, "b", "kb") == 1

    def test_temperature_c_to_f(self):
        tool = CalculatorTool()
        assert tool._convert_units(0, "c", "f") == 32

    def test_temperature_f_to_c(self):
        tool = CalculatorTool()
        assert tool._convert_units(32, "f", "c") == 0

    def test_temperature_k_to_c(self):
        tool = CalculatorTool()
        assert tool._convert_units(273.15, "k", "c") == 0

    def test_unknown_temperature_unit(self):
        tool = CalculatorTool()
        with pytest.raises(ValueError):
            tool._convert_temperature(1, "x", "c")

    def test_incompatible_units(self):
        tool = CalculatorTool()
        with pytest.raises(ValueError):
            tool._convert_units(1, "m", "kg")

    async def test_execute_expression_success_int(self):
        tool = CalculatorTool()
        result = await tool.execute(expression="2+2")
        assert result.success is True
        assert result.data == 4
        assert result.metadata["operation"] == "evaluate"

    async def test_execute_expression_success_float(self):
        tool = CalculatorTool()
        result = await tool.execute(expression="1/3")
        assert result.success is True
        assert isinstance(result.data, float)

    async def test_execute_convert_success(self):
        tool = CalculatorTool()
        result = await tool.execute(convert={"value": 100, "from": "cm", "to": "m"})
        assert result.success is True
        assert result.data == 1
        assert result.metadata["operation"] == "convert"

    async def test_execute_value_error(self):
        tool = CalculatorTool()
        result = await tool.execute(expression="2 ** 1001")
        assert result.success is False
        assert "Exponent too large" in result.error

    async def test_execute_zero_division(self):
        tool = CalculatorTool()
        result = await tool.execute(expression="1/0")
        assert result.success is False
        assert result.error == "Division by zero"

    async def test_execute_overflow(self):
        tool = CalculatorTool()
        result = await tool.execute(expression="exp(10000)")
        assert result.success is False
        assert result.error == "Result too large"

    async def test_execute_generic_exception(self):
        tool = CalculatorTool()
        result = await tool.execute(convert={"value": "x", "from": "m", "to": "cm"})
        assert result.success is False
        assert "could not convert string to float" in result.error

    def test_unsupported_constant_type(self):
        tool = CalculatorTool()
        with pytest.raises(ValueError, match="Unsupported constant type"):
            tool._safe_eval("'string'")

    def test_unknown_temperature_target_unit(self):
        tool = CalculatorTool()
        with pytest.raises(ValueError, match="Unknown temperature unit"):
            tool._convert_temperature(0, "c", "x")

    def test_validate_parameters_base_error_propagated(self):
        tool = CalculatorTool()
        error = tool.validate_parameters(unknown_param=123)
        assert "Unknown parameters" in error

    async def test_execute_float_converted_to_int(self):
        tool = CalculatorTool()
        result = await tool.execute(expression="4.0")
        assert result.success is True
        assert result.data == 4
        assert isinstance(result.data, int)

    async def test_execute_float_rounded(self):
        tool = CalculatorTool()
        result = await tool.execute(expression="1/3")
        assert result.success is True
        assert result.data == round(1 / 3, 10)

    async def test_execute_generic_exception_path(self, monkeypatch):
        tool = CalculatorTool()

        def boom(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(tool, "_safe_eval", boom)

        result = await tool.execute(expression="2+2")

        assert result.success is False
        assert result.error == "Calculation failed: boom"

    def test_unsupported_unary_operator(self):
        tool = CalculatorTool()
        node = ast.UnaryOp(op=ast.Not(), operand=ast.Constant(value=True))
        with pytest.raises(ValueError, match="Unsupported unary operator: Not"):
            tool._eval_node(node)


    def test_temperature_c_to_k(self):
        tool = CalculatorTool()
        assert tool._convert_temperature(0, "c", "k") == 273.15


    async def test_execute_convert_rounds_float(self):
        tool = CalculatorTool()
        result = await tool.execute(convert={"value": 1, "from": "in", "to": "m"})
        assert result.success is True
        assert result.data == round(0.0254, 10)
        assert isinstance(result.data, float)
