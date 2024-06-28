from sqlalchemy import create_engine, ForeignKey, Column, Integer, Float, Text
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
import numpy as np
import json


Base = declarative_base()


class Input(Base):
    __tablename__ = "inputs"
    id = Column(Integer, primary_key=True)
    radius = Column(Float)           # in [2.5; 5]
    thickness = Column(Float)        # in [0.03; 0.06]
    length = Column(Float)           # in [100; 200]
    mass_rna = Column(Integer)       # in [272 000, 820 000]
    sigma = Column(Float)            # in [235 000 000; 390 000 000]
    ship_velocity = Column(Float)    # in [0.5; 5]
    ship_mass = Column(Integer)      # in [6 000 000; 24 000 000]
    impact_point = Column(Float)     # in [0.25; 0.7]
    L_t = Column(Float)
    R_t = Column(Float)
    L_R = Column(Float)
    results = relationship('Output', back_populates='simu_input')


class Output(Base):
    __tablename__ = "outputs"
    id = Column(Integer, primary_key=True)
    time = Column(Text)
    force_x = Column(Text)
    force_y = Column(Text)
    force_z = Column(Text)
    delta = Column(Text)
    delta_loc = Column(Text)
    u_fowt = Column(Text)
    k_ship = Column(Text)
    v_ship = Column(Text)
    dynain = Column(Text)

    input_id = Column(Integer, ForeignKey('inputs.id'))
    simu_input = relationship('Input', back_populates='results')


def get_results(session, simu_id=None, simu_input=None):
    if simu_id:
        input_to_get = session.query(Input).filter(Input.id == simu_id).first()
    elif simu_input:
        input_to_get = simu_input
    else:
        print('Please, provide either a simulation ID or an input form the database.')
        return -1, -1

    if not input_to_get.results:
        print(f"There are no results for simulation: {input_to_get.id}")
        return -1, -1

    my_input = (input_to_get.radius, input_to_get.thickness, input_to_get.length, input_to_get.mass_rna,
                input_to_get.sigma, input_to_get.ship_velocity, input_to_get.ship_mass, input_to_get.impact_point)

    results = input_to_get.results[0]
    # print(results.id)
    time = json.loads(results.time)
    force_x = json.loads(results.force_x)
    delta = json.loads(results.delta)
    delta_loc = json.loads(results.delta_loc)
    u_fowt = json.loads(results.u_fowt)
    k_ship = json.loads(results.k_ship)

    force_x = [item / 10**6 for item in force_x]
    u_fowt = [item / 10 ** 6 for item in u_fowt]
    k_ship = [item / 10 ** 6 for item in k_ship]

    delta_glo = [d1 - d2 if d1 - d2 > 0 else 0 for d1, d2 in zip(delta, delta_loc)]

    _, index = min((value, idx) for idx, value in enumerate(k_ship))
    my_output = [time[:index], force_x[:index], delta[:index], delta_loc[:index], delta_glo[:index], u_fowt[:index], k_ship[:index]]

    my_output_int = [[i*0.05 for i in range(201)]]
    for output in my_output[1:]:
        _, out_interpolated = make_interpolation(my_output[0], output, nb_point=201)
        my_output_int.append(out_interpolated)

    for output in my_output[2:]:
        while len(output) < 201:
            output.append(output[-1])

    while len(my_output[1]) < 201:
        my_output[1].append(0)

    my_output[0] = [i*0.05 for i in range(201)]

    for output in my_output_int[2:]:
        while len(output) < 201:
            output.append(output[-1])

    while len(my_output_int[1]) < 201:
        my_output_int[1].append(0)

    return my_input, my_output, my_output_int


def get_session():
    engine = create_engine("sqlite:///instance/Simulations.db")

    Base.metadata.create_all(engine)

    session = sessionmaker(bind=engine)
    return session()


def get_data(session):
    return session.query(Input).all()


def delete_input(session, simu_id=None, simu_input=None):
    if simu_id:
        input_to_delete = session.query(Input).filter(Input.id == simu_id).first()
    elif simu_input:
        input_to_delete = simu_input
    else:
        print('Please, provide either a simulation ID or an input form the database.')
        return

    session.delete(input_to_delete)
    session.commit()
    return


def make_interpolation(x, y, nb_point=100, deg=10):
    # Quadratic interpolation
    coefficients = np.polyfit(np.array(x), np.array(y), deg=deg)
    quadratic_poly = np.poly1d(coefficients)

    # Generate points for the interpolated curve
    x_interpolated = np.linspace(min(x), max(x), nb_point)
    y_interpolated = quadratic_poly(x_interpolated)
    x_interpolated = [max(0, x) for x in x_interpolated]
    y_interpolated = [max(0, x) for x in y_interpolated]
    return x_interpolated, y_interpolated
