/**
 * TetrisPiece - Tetromino piece definitions
 * EXACT port from TetrisPiece.java in the RL-Glue Tetris environment
 *
 * Copyright 2007 Brian Tanner
 * Licensed under the Apache License, Version 2.0
 */

export class TetrisPiece {
    constructor() {
        // thePiece[orientation][row][col] - 4 orientations, 5x5 grid each
        this.thePiece = [
            [[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]],
            [[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]],
            [[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]],
            [[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]
        ];
        this.currentOrientation = 0;
    }

    /**
     * Set shape for a given orientation - EXACT from Java
     */
    setShape(direction, row0, row1, row2, row3, row4) {
        this.thePiece[direction][0] = row0.slice();
        this.thePiece[direction][1] = row1.slice();
        this.thePiece[direction][2] = row2.slice();
        this.thePiece[direction][3] = row3.slice();
        this.thePiece[direction][4] = row4.slice();
    }

    /**
     * Get shape for orientation - EXACT from Java
     */
    getShape(whichOrientation) {
        return this.thePiece[whichOrientation];
    }

    /**
     * Make Square piece (O) - EXACT from Java
     */
    static makeSquare() {
        const newPiece = new TetrisPiece();

        // Orientation 0,1,2,3 - all same
        const row0 = [0,0,0,0,0];
        const row1 = [0,0,1,1,0];
        const row2 = [0,0,1,1,0];
        const row3 = [0,0,0,0,0];
        const row4 = [0,0,0,0,0];
        newPiece.setShape(0, row0, row1, row2, row3, row4);
        newPiece.setShape(1, row0, row1, row2, row3, row4);
        newPiece.setShape(2, row0, row1, row2, row3, row4);
        newPiece.setShape(3, row0, row1, row2, row3, row4);

        return newPiece;
    }

    /**
     * Make T piece (Tri) - EXACT from Java
     */
    static makeTri() {
        const newPiece = new TetrisPiece();

        {
            // Orientation 0
            const row0 = [0,0,0,0,0];
            const row1 = [0,0,1,0,0];
            const row2 = [0,1,1,1,0];
            const row3 = [0,0,0,0,0];
            const row4 = [0,0,0,0,0];
            newPiece.setShape(0, row0, row1, row2, row3, row4);
        }
        {
            // Orientation 1
            const row0 = [0,0,0,0,0];
            const row1 = [0,0,1,0,0];
            const row2 = [0,0,1,1,0];
            const row3 = [0,0,1,0,0];
            const row4 = [0,0,0,0,0];
            newPiece.setShape(1, row0, row1, row2, row3, row4);
        }
        {
            // Orientation 2
            const row0 = [0,0,0,0,0];
            const row1 = [0,0,0,0,0];
            const row2 = [0,1,1,1,0];
            const row3 = [0,0,1,0,0];
            const row4 = [0,0,0,0,0];
            newPiece.setShape(2, row0, row1, row2, row3, row4);
        }
        {
            // Orientation 3
            const row0 = [0,0,0,0,0];
            const row1 = [0,0,1,0,0];
            const row2 = [0,1,1,0,0];
            const row3 = [0,0,1,0,0];
            const row4 = [0,0,0,0,0];
            newPiece.setShape(3, row0, row1, row2, row3, row4);
        }

        return newPiece;
    }

    /**
     * Make Line piece (I) - EXACT from Java
     */
    static makeLine() {
        const newPiece = new TetrisPiece();

        {
            // Orientation 0+2
            const row0 = [0,0,1,0,0];
            const row1 = [0,0,1,0,0];
            const row2 = [0,0,1,0,0];
            const row3 = [0,0,1,0,0];
            const row4 = [0,0,0,0,0];
            newPiece.setShape(0, row0, row1, row2, row3, row4);
            newPiece.setShape(2, row0, row1, row2, row3, row4);
        }
        {
            // Orientation 1+3
            const row0 = [0,0,0,0,0];
            const row1 = [0,0,0,0,0];
            const row2 = [0,1,1,1,1];
            const row3 = [0,0,0,0,0];
            const row4 = [0,0,0,0,0];
            newPiece.setShape(1, row0, row1, row2, row3, row4);
            newPiece.setShape(3, row0, row1, row2, row3, row4);
        }

        return newPiece;
    }

    /**
     * Make S piece - EXACT from Java
     */
    static makeSShape() {
        const newPiece = new TetrisPiece();

        {
            // Orientation 0+2
            const row0 = [0,0,0,0,0];
            const row1 = [0,1,0,0,0];
            const row2 = [0,1,1,0,0];
            const row3 = [0,0,1,0,0];
            const row4 = [0,0,0,0,0];
            newPiece.setShape(0, row0, row1, row2, row3, row4);
            newPiece.setShape(2, row0, row1, row2, row3, row4);
        }
        {
            // Orientation 1+3
            const row0 = [0,0,0,0,0];
            const row1 = [0,0,1,1,0];
            const row2 = [0,1,1,0,0];
            const row3 = [0,0,0,0,0];
            const row4 = [0,0,0,0,0];
            newPiece.setShape(1, row0, row1, row2, row3, row4);
            newPiece.setShape(3, row0, row1, row2, row3, row4);
        }

        return newPiece;
    }

    /**
     * Make Z piece - EXACT from Java
     */
    static makeZShape() {
        const newPiece = new TetrisPiece();

        {
            // Orientation 0+2
            const row0 = [0,0,0,0,0];
            const row1 = [0,0,1,0,0];
            const row2 = [0,1,1,0,0];
            const row3 = [0,1,0,0,0];
            const row4 = [0,0,0,0,0];
            newPiece.setShape(0, row0, row1, row2, row3, row4);
            newPiece.setShape(2, row0, row1, row2, row3, row4);
        }
        {
            // Orientation 1+3
            const row0 = [0,0,0,0,0];
            const row1 = [0,1,1,0,0];
            const row2 = [0,0,1,1,0];
            const row3 = [0,0,0,0,0];
            const row4 = [0,0,0,0,0];
            newPiece.setShape(1, row0, row1, row2, row3, row4);
            newPiece.setShape(3, row0, row1, row2, row3, row4);
        }

        return newPiece;
    }

    /**
     * Make L piece - EXACT from Java
     */
    static makeLShape() {
        const newPiece = new TetrisPiece();

        {
            // Orientation 0
            const row0 = [0,0,0,0,0];
            const row1 = [0,0,1,0,0];
            const row2 = [0,0,1,0,0];
            const row3 = [0,0,1,1,0];
            const row4 = [0,0,0,0,0];
            newPiece.setShape(0, row0, row1, row2, row3, row4);
        }
        {
            // Orientation 1
            const row0 = [0,0,0,0,0];
            const row1 = [0,0,0,0,0];
            const row2 = [0,1,1,1,0];
            const row3 = [0,1,0,0,0];
            const row4 = [0,0,0,0,0];
            newPiece.setShape(1, row0, row1, row2, row3, row4);
        }
        {
            // Orientation 2
            const row0 = [0,0,0,0,0];
            const row1 = [0,1,1,0,0];
            const row2 = [0,0,1,0,0];
            const row3 = [0,0,1,0,0];
            const row4 = [0,0,0,0,0];
            newPiece.setShape(2, row0, row1, row2, row3, row4);
        }
        {
            // Orientation 3
            const row0 = [0,0,0,0,0];
            const row1 = [0,0,0,1,0];
            const row2 = [0,1,1,1,0];
            const row3 = [0,0,0,0,0];
            const row4 = [0,0,0,0,0];
            newPiece.setShape(3, row0, row1, row2, row3, row4);
        }

        return newPiece;
    }

    /**
     * Make J piece - EXACT from Java
     */
    static makeJShape() {
        const newPiece = new TetrisPiece();

        {
            // Orientation 0
            const row0 = [0,0,0,0,0];
            const row1 = [0,0,1,0,0];
            const row2 = [0,0,1,0,0];
            const row3 = [0,1,1,0,0];
            const row4 = [0,0,0,0,0];
            newPiece.setShape(0, row0, row1, row2, row3, row4);
        }
        {
            // Orientation 1
            const row0 = [0,0,0,0,0];
            const row1 = [0,1,0,0,0];
            const row2 = [0,1,1,1,0];
            const row3 = [0,0,0,0,0];
            const row4 = [0,0,0,0,0];
            newPiece.setShape(1, row0, row1, row2, row3, row4);
        }
        {
            // Orientation 2
            const row0 = [0,0,0,0,0];
            const row1 = [0,0,1,1,0];
            const row2 = [0,0,1,0,0];
            const row3 = [0,0,1,0,0];
            const row4 = [0,0,0,0,0];
            newPiece.setShape(2, row0, row1, row2, row3, row4);
        }
        {
            // Orientation 3
            const row0 = [0,0,0,0,0];
            const row1 = [0,0,0,0,0];
            const row2 = [0,1,1,1,0];
            const row3 = [0,0,0,1,0];
            const row4 = [0,0,0,0,0];
            newPiece.setShape(3, row0, row1, row2, row3, row4);
        }

        return newPiece;
    }

    /**
     * String representation for debugging
     */
    toString() {
        let shapeBuffer = '';
        const shape = this.thePiece[this.currentOrientation];
        for (let i = 0; i < shape.length; i++) {
            for (let j = 0; j < shape[i].length; j++) {
                shapeBuffer += ' ' + shape[i][j];
            }
            shapeBuffer += '\n';
        }
        return shapeBuffer;
    }
}
