import argparse
from cli import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Herramienta RCWA de Sergio Montoya")
    subparsers = parser.add_subparsers(dest="command", help="Comandos disponibles")

    # Subcomando: simulate
    sim_parser = subparsers.add_parser('simulate', help='Ejecuta simulaciones RCWA')
    sim_parser.add_argument('-r', '--rangulo', action='store_true', help="Calcula frecuencia respecto al ángulo")
    sim_parser.add_argument('-t', '--thickness', action='store_true', help="Usa grosor del CSV")
    sim_parser.add_argument('-f', '--formula', type=str, nargs='+', help="Filtra por fórmula")
    sim_parser.add_argument('-i', '--information', type=str, default="./dielectricos_absolutamente_todo.csv")

    # Subcomando: report
    rep_parser = subparsers.add_parser('report', help='Genera reporte LaTeX e imágenes')
    rep_parser.add_argument('-p', '--pattern', type=str, default=r'.*\.pkl$', help="Regex para archivos pkl")
    rep_parser.add_argument('--rangle', action='store_true', help="Procesar datos de rotación")

    args = parser.parse_args()

    if args.command == 'simulate':
        handle_simulate(args)
    elif args.command == 'report':
        handle_report(args)
    else:
        parser.print_help()
