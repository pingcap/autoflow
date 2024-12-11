from sqlmodel import Session, SQLModel, select


class BaseRepo:
    model_cls: SQLModel

    def get(self, session: Session, id: int):
        return session.get(self.model_cls, id)

    def get_all(self, session: Session):
        return session.exec(select(self.model_cls)).all()

    def create(self, session: Session, obj: SQLModel):
        session.add(obj)
        session.commit()
        session.refresh(obj)
        return obj
