from peewee import SqliteDatabase, Model, CharField, InternalError, ForeignKeyField, \
    DateTimeField, IntegerField, SmallIntegerField, FloatField, TimeField, AutoField
import datetime

db = SqliteDatabase('./models/HeroAI.db')


class Models(Model):
    ID = AutoField()
    CreationDate = DateTimeField(default=datetime.datetime.now())
    Version = SmallIntegerField(null=True, unique=True)

    class Meta:
        database = db


class ModelFeatures(Model):
    Model = ForeignKeyField(Models, backref="Features")
    # Current accuracy and loss for the model
    TrainAccuracy = FloatField()
    TrainLoss = FloatField()
    ValidationAccuracy = FloatField()
    ValidationLoss = FloatField()
    EvaluationAccuracy = FloatField()
    EvaluationLoss = FloatField()
    TrainTime = TimeField()

    # Model permanent features
    BUFFER_SIZE = IntegerField()
    BATCH_SIZE = SmallIntegerField()
    EPOCHS = SmallIntegerField()
    VALIDATION_STEPS = SmallIntegerField()
    LEARNING_RATE = FloatField()
    EMBED_DIM = SmallIntegerField()
    DEEP_UNITS = SmallIntegerField()
    DENSE_UNITS = SmallIntegerField()
    DROPOUT = FloatField()
    MAX_FEATURES = IntegerField()
    MAX_LENGTH = SmallIntegerField()
    TRAIN_TAKE_SIZE = IntegerField()
    TEST_TAKE_SIZE = IntegerField()

    class Meta:
        database = db


class ModelFiles(Model):
    ID = AutoField()
    Model = ForeignKeyField(Models, backref="Files")
    FileType = CharField(max_length=25)
    Path = CharField(max_length=255, unique=True)

    class Meta:
        database = db


def newModel():
    try:
        model = Models.create()
        if model:
            return model
        return None
    except Exception as e:
        print("Error saving model to database - " + str(e))
        return None


def addFeatures(model: Models, t_accuracy: float, t_loss: float, v_accuracy: float, v_loss: float, e_accuracy: float,
                     e_loss: float, tr_time, buffer_size, batch_size, epochs, validation_steps, learning_rate, dropout,
                     embed_dim, deep_units, dense_units, max_features, max_length, train_take_size, test_take_size):
    if model:
        features = (ModelFeatures
                    .insert(Model=model, TrainAccuracy=t_accuracy, TrainLoss=t_loss, ValidationAccuracy=v_accuracy,
                            ValidationLoss=v_loss, EvaluationAccuracy=e_accuracy, EvaluationLoss=e_loss,
                            TrainTime=tr_time, BUFFER_SIZE=buffer_size, BATCH_SIZE=batch_size, EPOCHS=epochs,
                            VALIDATION_STEPS=validation_steps, LEARNING_RATE=learning_rate, EMBED_DIM=embed_dim,
                            DEEP_UNITS=deep_units, DENSE_UNITS=dense_units, MAX_FEATURES=max_features, DROPOUT=dropout,
                            MAX_LENGTH=max_length, TRAIN_TAKE_SIZE=train_take_size, TEST_TAKE_SIZE=test_take_size)
                    .on_conflict_replace()
                    .execute())
        return features


def addFile(model, filepath: str, filetype="SaveModel"):
    if model:
        file = (ModelFiles
                .insert(Model=model, FileType=filetype, Path=filepath)
                .on_conflict_replace()
                .execute())
        return file


def getNewestModel(dev=False):
    if not dev:
        print("get model by version")
        query = Models.select().join(ModelFeatures).switch(Models).join(ModelFiles).switch(Models).order_by(Models.Version.desc())
        if query.exists():
            if query[0].Version:
                return query[0]
        print("no models with a version, getting model by creation date")
    query = Models.select().join(ModelFeatures).switch(Models).join(ModelFiles).switch(Models).order_by(Models.CreationDate.desc())
    if query.exists():
        return query[0]


def getModelFile(model: Models, file_type="SaveModel"):
    files = model.Files
    file = [file for file in files if file.FileType == file_type]
    if file:
        return file[0]
    return None


def create_tables():
    with db:
        db.create_tables([Models, ModelFeatures, ModelFiles])


create_tables()
