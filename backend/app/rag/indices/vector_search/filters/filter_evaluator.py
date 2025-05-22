from typing import Any, Optional, Union
from sqlalchemy import and_, or_, func
from sqlalchemy.sql.elements import BinaryExpression
from llama_index.core.vector_stores.types import (
    MetadataFilters,
    FilterOperator,
    FilterCondition,
)
import logging

logger = logging.getLogger(__name__)


class FilterEvaluator:
    """Generic filter evaluator that supports both SQL condition building and in-memory filtering"""
    
    # Define operators that support numeric comparisons
    NUMERIC_OPERATORS = {
        FilterOperator.GT,  # int, float
        FilterOperator.LT,  # int, float
        FilterOperator.GTE,  # int, float
        FilterOperator.LTE,  # int, float
    }
    
    # Define operators that support array operations
    ARRAY_OPERATORS = {
        FilterOperator.IN,  # string or number
        FilterOperator.NIN,  # string or number
        FilterOperator.ANY,  # array of strings
        FilterOperator.ALL,  # array of strings
        FilterOperator.CONTAINS,  # string or number
    }
    
    # Define operators that support text operations
    TEXT_OPERATORS = {
        FilterOperator.TEXT_MATCH,  # string
        FilterOperator.TEXT_MATCH_INSENSITIVE,  # string
    }
    
    # Define operators that support basic comparisons (string, int, float)
    BASIC_OPERATORS = {
        FilterOperator.EQ,  # string, int, float
        FilterOperator.NE,  # string, int, float
    }
    
    @staticmethod
    def _validate_numeric_operator(operator: FilterOperator, value: Any) -> bool:
        """Validate the value type for numeric operators"""
        if not isinstance(value, (int, float)):
            logger.error(f"Operator {operator} requires numeric value (int or float), got {type(value)}")
            return False
        return True
    
    @staticmethod
    def _validate_array_operator(operator: FilterOperator, value: Any) -> bool:
        """Validate the value type for array operators"""
        if operator in [FilterOperator.ANY, FilterOperator.ALL]:
            # ANY and ALL operators require array of strings
            if not isinstance(value, (list, tuple)):
                logger.error(f"Operator {operator} requires array of strings, got {type(value)}")
                return False
            if not all(isinstance(item, str) for item in value):
                logger.error(f"Operator {operator} requires array of strings, got array with non-string items")
                return False
        else:
            # IN, NIN, CONTAINS operators accept string or number
            if not isinstance(value, (list, tuple)):
                logger.error(f"Operator {operator} requires array value, got {type(value)}")
                return False
            if not all(isinstance(item, (str, int, float)) for item in value):
                logger.error(f"Operator {operator} requires array of strings or numbers, got array with invalid items")
                return False
        return True
    
    @staticmethod
    def _validate_text_operator(operator: FilterOperator, value: Any) -> bool:
        """Validate the value type for text operators"""
        if not isinstance(value, str):
            logger.error(f"Operator {operator} requires string value, got {type(value)}")
            return False
        return True
    
    @staticmethod
    def _validate_basic_operator(operator: FilterOperator, value: Any) -> bool:
        """Validate the value type for basic comparison operators"""
        if not isinstance(value, (str, int, float)):
            logger.error(f"Operator {operator} requires string, int, or float value, got {type(value)}")
            return False
        return True
    
    @staticmethod
    def evaluate_filter(value: Any, filter_value: Any, operator: FilterOperator, 
                       sql_mode: bool = False, meta_column: Any = None) -> Union[bool, BinaryExpression]:
        """
        Evaluate a single filter condition
        
        Args:
            value: The value to compare against
            filter_value: The filter condition value
            operator: The filter operator
            sql_mode: Whether to return SQL condition
            meta_column: The metadata column reference in SQL mode
            
        Returns:
            SQL condition expression in SQL mode, boolean value otherwise
            Returns False for in-memory mode or None for SQL mode if an error occurs
        """
        try:
            if sql_mode and meta_column is None:
                logger.error("SQL mode requires meta_column parameter")
                return None

            # Validate operator
            if not isinstance(operator, FilterOperator):
                logger.error(f"Invalid operator type: {type(operator)}")
                return False if not sql_mode else None

            # Validate value types based on operator
            if operator in FilterEvaluator.NUMERIC_OPERATORS:
                if not FilterEvaluator._validate_numeric_operator(operator, filter_value):
                    return False if not sql_mode else None
            elif operator in FilterEvaluator.ARRAY_OPERATORS:
                if not FilterEvaluator._validate_array_operator(operator, filter_value):
                    return False if not sql_mode else None
            elif operator in FilterEvaluator.TEXT_OPERATORS:
                if not FilterEvaluator._validate_text_operator(operator, filter_value):
                    return False if not sql_mode else None
            elif operator in FilterEvaluator.BASIC_OPERATORS:
                if not FilterEvaluator._validate_basic_operator(operator, filter_value):
                    return False if not sql_mode else None
            elif operator == FilterOperator.IS_EMPTY:
                # IS_EMPTY doesn't require value validation
                pass
            else:
                logger.error(f"Unsupported operator: {operator}")
                return False if not sql_mode else None

            # Operator logic processing
            if operator == FilterOperator.EQ:
                return meta_column == filter_value if sql_mode else value == filter_value
            elif operator == FilterOperator.NE:
                return meta_column != filter_value if sql_mode else value != filter_value
            elif operator == FilterOperator.GT:
                return meta_column > filter_value if sql_mode else value > filter_value
            elif operator == FilterOperator.GTE:
                return meta_column >= filter_value if sql_mode else value >= filter_value
            elif operator == FilterOperator.LT:
                return meta_column < filter_value if sql_mode else value < filter_value
            elif operator == FilterOperator.LTE:
                return meta_column <= filter_value if sql_mode else value <= filter_value
            elif operator == FilterOperator.IN:
                return meta_column.in_(filter_value) if sql_mode else value in filter_value
            elif operator == FilterOperator.NIN:
                return ~meta_column.in_(filter_value) if sql_mode else value not in filter_value
            elif operator == FilterOperator.CONTAINS:
                return meta_column.contains(filter_value) if sql_mode else any(item in value for item in filter_value)
            elif operator == FilterOperator.IS_EMPTY:
                if sql_mode:
                    return or_(
                        meta_column.is_(None),
                        meta_column == "",
                        meta_column == "[]"
                    )
                return value is None or value == "" or (isinstance(value, list) and len(value) == 0)
            elif operator == FilterOperator.TEXT_MATCH:
                if sql_mode:
                    return meta_column.contains(str(filter_value))
                return str(filter_value) in str(value)
            elif operator == FilterOperator.TEXT_MATCH_INSENSITIVE:
                if sql_mode:
                    return func.lower(meta_column).contains(func.lower(str(filter_value)))
                return str(filter_value).lower() in str(value).lower()
            elif operator == FilterOperator.ANY:
                if sql_mode:
                    any_conditions = [meta_column.contains(str(item)) for item in filter_value]
                    return or_(*any_conditions)
                return any(item in value for item in filter_value)
            elif operator == FilterOperator.ALL:
                if sql_mode:
                    all_conditions = [meta_column.contains(str(item)) for item in filter_value]
                    return and_(*all_conditions)
                return all(item in value for item in filter_value)
            else:
                logger.error(f"Unsupported operator: {operator}")
                return False if not sql_mode else None

        except Exception as e:
            logger.error(f"Error evaluating filter: {str(e)}")
            return False if not sql_mode else None

    @staticmethod
    def evaluate_conditions(filters: MetadataFilters, metadata: Optional[dict] = None) -> Optional[bool]:
        """
        Evaluate filter conditions for in-memory evaluation
        
        Args:
            filters: Collection of filter conditions
            metadata: The metadata dictionary to evaluate against
            
        Returns:
            Boolean value indicating if the conditions are met
            Returns None if an error occurs
        """
        try:
            if not filters.filters:
                return True

            conditions = []
            for f in filters.filters:
                if isinstance(f, MetadataFilters):
                    # Recursively process nested compound conditions
                    sub_condition = FilterEvaluator.evaluate_conditions(f, metadata)
                    if sub_condition is not None:
                        conditions.append(sub_condition)
                else:
                    # Get value from metadata
                    if metadata is None or f.key not in metadata:
                        continue
                    value = metadata[f.key]
                    condition = FilterEvaluator.evaluate_filter(
                        value, f.value, f.operator, False
                    )
                    if condition is not None:  # Only append if not None
                        conditions.append(condition)

            if not conditions:
                return None

            if filters.condition == FilterCondition.AND:
                return all(conditions)
            else:  # FilterCondition.OR
                return any(conditions)
        except Exception as e:
            logger.error(f"Error evaluating conditions: {str(e)}")
            return None
        
    @staticmethod
    def build_filter_conditions(filters: MetadataFilters, meta_column: Any) -> Optional[BinaryExpression]:
        """
        Build SQL filter conditions
        
        Args:
            filters: Collection of filter conditions
            meta_column: The metadata column reference
            
        Returns:
            SQL condition expression
            Returns None if an error occurs
        """
        try:
            if not filters.filters:
                return None

            conditions = []
            for f in filters.filters:
                if isinstance(f, MetadataFilters):
                    # Recursively process nested compound conditions
                    sub_condition = FilterEvaluator.build_filter_conditions(f, meta_column)
                    if sub_condition is not None:
                        conditions.append(sub_condition)
                else:
                    condition = FilterEvaluator.evaluate_filter(
                        None, f.value, f.operator, True, meta_column
                    )
                    if condition is not None:  # Only append if not None
                        conditions.append(condition)

            if not conditions:
                return None

            if filters.condition == FilterCondition.AND:
                return and_(*conditions)
            else:  # FilterCondition.OR
                return or_(*conditions)
        except Exception as e:
            logger.error(f"Error building filter conditions: {str(e)}")
            return None