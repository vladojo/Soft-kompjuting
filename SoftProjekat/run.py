from video_processor import process_video


def generate_results(student_id='RA 32/2015', student_name='Vladimir Jovanovic', results_path='data/results.txt'):
    """Runs solution and generates student's results."""
    f = open(results_path, "w+")
    f.write(f'{student_id} {student_name}\r')
    f.write('file	sum\r')

    for i in range(10):
        file_name = f'video-{i}.avi'
        file_path = f'data/videos/{file_name}'

        result = process_video(file_path)
        print(f'Processing file {file_name} ended. Result is: {result}')
        f.write(f'{file_name}\t{result}\r')

    f.close()


if __name__ == '__main__':
    generate_results()
