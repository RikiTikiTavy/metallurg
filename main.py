import imageio

from metallurg.consts import *
from metallurg.utils import *
from metallurg.net import get_metallurg_net
from metallurg.knn import create_knn
from metallurg.transforms import get_data_transform


def main():
    # загрузили модели
    model = get_metallurg_net(NET_FILENAME)
    model_ts = create_knn(KNN_FILENAME)

    # трансформации для нейронки
    data_transform = get_data_transform()

    # загрузили видео
    video_reader = imageio.get_reader(VIDEO_FILENAME)
    fps = get_fps(video_reader)
    total_frames = video_reader.get_length()

    # дату получаем один раз
    date = None

    # данные для csv-файлика
    csv_data = []

    # массив из N-кадров
    frames = []

    # проходимся по всем кадрам в видео
    for idx, frame in enumerate(video_reader):
        mod = idx % fps
        # учитываем первые N-кадров
        if mod < NUM_EVAL_FRAMES:
            # складываем кадры в массив
            frames.append(frame)
            # если наступил N-кадр или это последний кадр в видео
            if (mod == NUM_EVAL_FRAMES - 1) or (idx == total_frames - 1):
                # дату получаем один раз
                if date is None:
                    date = get_timestamp_roi(frames[-1], DATE_COORDS, DATE_THRESH, THRESH_BINARY)
                    date = parse_timestamp(model_ts, date, DATE_DIGITS, DATE_FORMAT)

                # считали время
                time = get_timestamp_roi(frames[-1], TIME_COORDS, TIME_THRESH, THRESH_BINARY_INV)
                time = parse_timestamp(model_ts, time, TIME_DIGITS, TIME_FORMAT)

                # получили timestamp
                timestamp = TIMESTAMP_FORMAT.format(date, time)

                # получили цифры в нужном для нейронки формате
                digits = get_digits(frames, data_transform)
                # получили предсказания по каждой цифре
                num_frames = len(frames)
                confidence, predicted = parse_digits(model, digits, num_frames)

                # распарсили все значения
                furnace     = get_readout(predicted, FURNACE_INDEXES)  # печь
                crucible    = get_readout(predicted, CRUCIBLE_INDEXES)  # тигель
                wire        = get_readout(predicted, WIRE_INDEXES, WIRE_FORMAT)  # проволока

                # сохранили данные
                csv_line = [timestamp, furnace, crucible, wire]
                csv_data.append(csv_line)

                # вывод в консоль
                print(RESULT_FORMAT.format(csv_line))
                
                # обнулили массив из N-кадров
                frames = []

    # сохранили результат в файл
    write_csv(csv_data, RESULT_FILENAME)


if __name__ == '__main__':
    main()
