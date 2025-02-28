from typing import (
    AbstractSet,
    Any,
    Callable,
    ClassVar,
    Dict,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from pydantic import BaseModel
from sqlalchemy.orm import (
    RelationshipProperty,
    declared_attr,
)
from sqlalchemy.orm.attributes import set_attribute
from sqlalchemy.orm.instrumentation import is_instrumented
from sqlalchemy.sql.schema import MetaData
from sqlmodel.main import (
    IS_PYDANTIC_V2,
    PYDANTIC_VERSION,
    SQLModelConfig,
    _calculate_keys,
    finish_init,
    get_model_fields,
    init_pydantic_private_attrs,
    is_table_model_class,
    sqlmodel_init,
    sqlmodel_validate,
    default_registry,
    _TSQLModel,
    IncEx,
)
from typing_extensions import Literal, deprecated
from autoflow.utils.sql_model_metaclass import PatchSQLModelMetaclass


class PatchSQLModel(
    BaseModel, metaclass=PatchSQLModelMetaclass, registry=default_registry
):
    # SQLAlchemy needs to set weakref(s), Pydantic will set the other slots values
    __slots__ = ("__weakref__",)
    __tablename__: ClassVar[Union[str, Callable[..., str]]]
    __sqlmodel_relationships__: ClassVar[Dict[str, RelationshipProperty[Any]]]
    __name__: ClassVar[str]
    metadata: ClassVar[MetaData]
    __allow_unmapped__ = True  # https://docs.sqlalchemy.org/en/20/changelog/migration_20.html#migration-20-step-six

    if IS_PYDANTIC_V2:
        model_config = SQLModelConfig(from_attributes=True)
    else:

        class Config:
            orm_mode = True

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        new_object = super().__new__(cls)
        # SQLAlchemy doesn't call __init__ on the base class when querying from DB
        # Ref: https://docs.sqlalchemy.org/en/14/orm/constructors.html
        # Set __fields_set__ here, that would have been set when calling __init__
        # in the Pydantic model so that when SQLAlchemy sets attributes that are
        # added (e.g. when querying from DB) to the __fields_set__, this already exists
        init_pydantic_private_attrs(new_object)
        return new_object

    def __init__(__pydantic_self__, **data: Any) -> None:
        # Uses something other than `self` the first arg to allow "self" as a
        # settable attribute

        # SQLAlchemy does very dark black magic and modifies the __init__ method in
        # sqlalchemy.orm.instrumentation._generate_init()
        # so, to make SQLAlchemy work, it's needed to explicitly call __init__ to
        # trigger all the SQLAlchemy logic, it doesn't work using cls.__new__, setting
        # attributes obj.__dict__, etc. The __init__ method has to be called. But
        # there are cases where calling all the default logic is not ideal, e.g.
        # when calling Model.model_validate(), as the validation is done outside
        # of instance creation.
        # At the same time, __init__ is what users would normally call, by creating
        # a new instance, which should have validation and all the default logic.
        # So, to be able to set up the internal SQLAlchemy logic alone without
        # executing the rest, and support things like Model.model_validate(), we
        # use a contextvar to know if we should execute everything.
        if finish_init.get():
            sqlmodel_init(self=__pydantic_self__, data=data)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"_sa_instance_state"}:
            self.__dict__[name] = value
            return
        else:
            # Set in SQLAlchemy, before Pydantic to trigger events and updates
            if is_table_model_class(self.__class__) and is_instrumented(self, name):  # type: ignore[no-untyped-call]
                set_attribute(self, name, value)
            # Set in Pydantic model to trigger possible validation changes, only for
            # non relationship values
            if name not in self.__sqlmodel_relationships__:
                super().__setattr__(name, value)

    def __repr_args__(self) -> Sequence[Tuple[Optional[str], Any]]:
        # Don't show SQLAlchemy private attributes
        return [
            (k, v)
            for k, v in super().__repr_args__()
            if not (isinstance(k, str) and k.startswith("_sa_"))
        ]

    @declared_attr  # type: ignore
    def __tablename__(cls) -> str:
        return cls.__name__.lower()

    @classmethod
    def model_validate(
        cls: Type[_TSQLModel],
        obj: Any,
        *,
        strict: Union[bool, None] = None,
        from_attributes: Union[bool, None] = None,
        context: Union[Dict[str, Any], None] = None,
        update: Union[Dict[str, Any], None] = None,
    ) -> _TSQLModel:
        return sqlmodel_validate(
            cls=cls,
            obj=obj,
            strict=strict,
            from_attributes=from_attributes,
            context=context,
            update=update,
        )

    def model_dump(
        self,
        *,
        mode: Union[Literal["json", "python"], str] = "python",
        include: IncEx = None,
        exclude: IncEx = None,
        context: Union[Dict[str, Any], None] = None,
        by_alias: bool = False,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        round_trip: bool = False,
        warnings: Union[bool, Literal["none", "warn", "error"]] = True,
        serialize_as_any: bool = False,
    ) -> Dict[str, Any]:
        if PYDANTIC_VERSION >= "2.7.0":
            extra_kwargs: Dict[str, Any] = {
                "context": context,
                "serialize_as_any": serialize_as_any,
            }
        else:
            extra_kwargs = {}
        if IS_PYDANTIC_V2:
            return super().model_dump(
                mode=mode,
                include=include,
                exclude=exclude,
                by_alias=by_alias,
                exclude_unset=exclude_unset,
                exclude_defaults=exclude_defaults,
                exclude_none=exclude_none,
                round_trip=round_trip,
                warnings=warnings,
                **extra_kwargs,
            )
        else:
            return super().dict(
                include=include,
                exclude=exclude,
                by_alias=by_alias,
                exclude_unset=exclude_unset,
                exclude_defaults=exclude_defaults,
                exclude_none=exclude_none,
            )

    @deprecated(
        """
        🚨 `obj.dict()` was deprecated in SQLModel 0.0.14, you should
        instead use `obj.model_dump()`.
        """
    )
    def dict(
        self,
        *,
        include: IncEx = None,
        exclude: IncEx = None,
        by_alias: bool = False,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
    ) -> Dict[str, Any]:
        return self.model_dump(
            include=include,
            exclude=exclude,
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
        )

    @classmethod
    @deprecated(
        """
        🚨 `obj.from_orm(data)` was deprecated in SQLModel 0.0.14, you should
        instead use `obj.model_validate(data)`.
        """
    )
    def from_orm(
        cls: Type[_TSQLModel], obj: Any, update: Optional[Dict[str, Any]] = None
    ) -> _TSQLModel:
        return cls.model_validate(obj, update=update)

    @classmethod
    @deprecated(
        """
        🚨 `obj.parse_obj(data)` was deprecated in SQLModel 0.0.14, you should
        instead use `obj.model_validate(data)`.
        """
    )
    def parse_obj(
        cls: Type[_TSQLModel], obj: Any, update: Optional[Dict[str, Any]] = None
    ) -> _TSQLModel:
        if not IS_PYDANTIC_V2:
            obj = cls._enforce_dict_if_root(obj)  # type: ignore[attr-defined] # noqa
        return cls.model_validate(obj, update=update)

    # From Pydantic, override to only show keys from fields, omit SQLAlchemy attributes
    @deprecated(
        """
        🚨 You should not access `obj._calculate_keys()` directly.

        It is only useful for Pydantic v1.X, you should probably upgrade to
        Pydantic v2.X.
        """,
        category=None,
    )
    def _calculate_keys(
        self,
        include: Optional[Mapping[Union[int, str], Any]],
        exclude: Optional[Mapping[Union[int, str], Any]],
        exclude_unset: bool,
        update: Optional[Dict[str, Any]] = None,
    ) -> Optional[AbstractSet[str]]:
        return _calculate_keys(
            self,
            include=include,
            exclude=exclude,
            exclude_unset=exclude_unset,
            update=update,
        )

    def sqlmodel_update(
        self: _TSQLModel,
        obj: Union[Dict[str, Any], BaseModel],
        *,
        update: Union[Dict[str, Any], None] = None,
    ) -> _TSQLModel:
        use_update = (update or {}).copy()
        if isinstance(obj, dict):
            for key, value in {**obj, **use_update}.items():
                if key in get_model_fields(self):
                    setattr(self, key, value)
        elif isinstance(obj, BaseModel):
            for key in get_model_fields(obj):
                if key in use_update:
                    value = use_update.pop(key)
                else:
                    value = getattr(obj, key)
                setattr(self, key, value)
            for remaining_key in use_update:
                if remaining_key in get_model_fields(self):
                    value = use_update.pop(remaining_key)
                    setattr(self, remaining_key, value)
        else:
            raise ValueError(
                "Can't use sqlmodel_update() with something that "
                f"is not a dict or SQLModel or Pydantic model: {obj}"
            )
        return self
